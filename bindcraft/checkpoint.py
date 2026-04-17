"""
BindCraft v2 - 训练过程 Checkpoint 模块

每个优化步骤保存一次：
  1. 完整复合物 PDB 文件（binder + target 的全原子坐标）
  2. JSON 元数据（序列 / loss / confidence / ...）
  3. JSON 原子级 pLDDT（用于可视化着色）

设计原则：
  - 独立模块，不侵入主 pipeline 逻辑
  - 通过 ZerothOrderResult + atom_array 信息构建 PDB
  - 所有文件写入同一输出目录，便于统一管理
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import biotite

# 从 protenix 常量中获取 ATOM14 / ATOM37 / ONE_TO_THREE 映射
from protenix.data.constants import ATOM14, ATOM37_ORDER

# =============================================================================
# 氨基酸单字母 ↔ 三字母映射
# =============================================================================

AA_1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS",
    "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR",
    "V": "VAL",
}
AA_3_TO_1 = {v: k for k, v in AA_1_TO_3.items()}

# Protenix 中 binder 固定在 chain_id=0，target 在 chain_id=1
# 但 biotite 的 atom_array.chain_id 是字符串 'A' / 'B'
CHAIN_BINDER_STR = "A"
CHAIN_TARGET_STR = "B"


# =============================================================================
# 核心：atom37 dense_index → 标准 PDB ATOM 行写入
# =============================================================================

def _dense_index_to_atom37_atomname(
    dense_index: int, resname: str
) -> Optional[str]:
    """
    给定残基名和三字母代码，返回该残基在 atom37 体系中的原子名。

    atom37 的定义是按固定顺序排列的 37 种原子类型（见 constants.ATOM37）。
    这里的 dense_index 对应 atom37 中该原子所在的槽位。
    如果该残基在这个槽位没有对应原子，返回 None。

    我们通过 ATOM14 集合来反查：遍历该残基的 ATOM14 原子，
    用 ATOM37_ORDER 查到它们在 atom37 中的索引，与 dense_index 比对。
    """
    atom_names = ATOM14.get(resname, ())
    for name in atom_names:
        if ATOM37_ORDER.get(name, -1) == dense_index:
            return name
    return None


# =============================================================================
# 从 SampleDict 重建 atom_array（用于获取原子/残基元信息）
# =============================================================================

def _rebuild_atom_array(
    binder_seq: str,
    target_seq: Any,
) -> "biotite.AtomArray":            # type hint forward-ref
    """
    通过 SampleDict → SampleDictToFeatures 路径重建 atom_array。
    与 build_complex_features 中的构建逻辑完全一致，
    只是额外保留 AtomArray 供 checkpoint 使用。
    """
    from protenix.data.inference.json_to_feature import SampleDictToFeatures
    from protenix.data.inference.infer_dataloader import InferenceTemplateFeaturizer
    from protenix.data.utils import make_dummy_feature

    def _normalize_seq(seq: Any) -> str:
        # Some upstream callers may pass ["ACDE..."] instead of "ACDE...".
        # Protenix expects a single string of 1-letter residue codes.
        if isinstance(seq, str):
            return seq
        if isinstance(seq, (list, tuple)):
            if len(seq) == 1 and isinstance(seq[0], str):
                return seq[0]
            if all(isinstance(x, str) and len(x) == 1 for x in seq):
                return "".join(seq)
        raise TypeError(f"sequence must be str (or list/tuple convertible to str), got {type(seq)}")

    def _normalize_chain_seqs(seq: Any) -> List[str]:
        # Accept either a single chain "ACDE..." or multi-chain ["...","..."].
        if isinstance(seq, str):
            return [seq]
        if isinstance(seq, (list, tuple)):
            if len(seq) == 0:
                raise TypeError("target sequence list cannot be empty")
            # ["ACDE..."] or ["ACDE...","FGH..."]
            if all(isinstance(x, str) and len(x) >= 1 for x in seq):
                # If it's actually a list of single-letter chars, merge.
                if all(len(x) == 1 for x in seq):
                    return ["".join(seq)]
                return list(seq)
        raise TypeError(f"target_seq must be str or list/tuple[str], got {type(seq)}")

    binder_seq = _normalize_seq(binder_seq)
    target_chain_seqs = _normalize_chain_seqs(target_seq)

    sample_dict = {
        "sequences": (
            [{"proteinChain": {"sequence": binder_seq, "count": 1}}]
            + [{"proteinChain": {"sequence": s, "count": 1}} for s in target_chain_seqs]
        ),
        "name": "checkpoint_complex",
    }

    sample2feat = SampleDictToFeatures(sample_dict)
    features_dict, atom_array, _ = sample2feat.get_feature_dict()

    # template（这里不用 MSA/template，只保留 AtomArray）
    _ = InferenceTemplateFeaturizer.make_template_feature(
        bioassembly=sample_dict["sequences"],
        atom_array=atom_array,
        use_template=False,
        online_template_featurizer=None,
    )
    dummy_feats = ["msa"]
    if len(_) == 0:
        dummy_feats.append("template")
    features_dict = make_dummy_feature(features_dict=features_dict, dummy_feats=dummy_feats)

    return atom_array


# =============================================================================
# 坐标类型转换辅助
# =============================================================================

def _to_numpy(tensor_or_array) -> np.ndarray:
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy()
    return np.asarray(tensor_or_array)


# =============================================================================
# PDB 写入
# =============================================================================

def write_pdb(
    output_path: str,
    atom_array,                   # biotite AtomArray，带 atom_name / res_name / chain_id / res_id
    coordinates: np.ndarray,      # shape [N_atom, 3]，float
    b_factors: Optional[np.ndarray] = None,  # shape [N_atom]，可选
    plddt_per_atom: Optional[np.ndarray] = None,  # shape [N_atom]，可选（直接写入 b_factor）
) -> None:
    """
    将完整复合物坐标写入标准 PDB 文件。

    与 Protenix 自带的 save_structure_cif 不同，这里写入标准 PDB 格式，
    兼容 PyMOL / ChimeraX / AlphaFold ColorbyConf 等工具链。

    Args:
        output_path:   输出 .pdb 路径
        atom_array:    biotite AtomArray，从 _rebuild_atom_array 获取
        coordinates:   预测坐标，[N_atom, 3]
        b_factors:     每个原子的 B-factor（可选）
        plddt_per_atom: 每个原子的 pLDDT 分数（0~100），会覆写到 b_factor 列
    """
    if plddt_per_atom is not None:
        b_factors = plddt_per_atom

    coords = _to_numpy(coordinates)
    if b_factors is not None:
        b_vals = _to_numpy(b_factors)
    else:
        b_vals = np.zeros(coords.shape[0])

    N_atom = coords.shape[0]
    assert atom_array.array_length() == N_atom, (
        f"atom_array length {atom_array.array_length()} != coordinates {N_atom}"
    )

    atom_names = atom_array.atom_name   # shape [N_atom]
    res_names  = atom_array.res_name   # shape [N_atom]
    chain_ids   = atom_array.chain_id   # shape [N_atom]
    res_ids     = atom_array.res_id     # shape [N_atom]
    elements    = atom_array.element    # shape [N_atom]

    lines: List[str] = []
    for i in range(N_atom):
        x, y, z = coords[i]
        atom_name = str(atom_names[i]).strip()
        res_name  = str(res_names[i]).strip()
        chain_id  = str(chain_ids[i]).strip()
        if not chain_id:
            chain_id = "A"
        res_id = int(res_ids[i]) if res_ids[i] is not None else 1
        icode  = " "   # insertion code，暂不使用
        b_val  = float(b_vals[i]) if i < len(b_vals) else 0.0
        elem   = str(elements[i]).strip()
        if not elem:
            elem = atom_name[0] if len(atom_name) > 0 else "C"

        occ = 1.0
        line = (
            f"ATOM  {i+1:5d}  {atom_name:<4s}{icode} {res_name:<3s} {chain_id}"
            f"{res_id:4d}{icode}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{occ:6.2f}{b_val:6.2f}"
            f"          {elem:>2s}"
        )
        lines.append(line)

    header = [
        "REMARK  1  Generated by BindCraft Checkpoint Module",
        "REMARK  1  Complex: Binder + Target",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(lines) + "\n")
        f.write("END\n")


# =============================================================================
# JSON 元数据写入
# =============================================================================

def write_metadata_json(
    output_path: str,
    step: int,
    beam_id: int,
    binder_seq: str,
    target_seq: Any,
    total_loss: float,
    loss_dict: Dict[str, float],
    summary_confidence: Dict[str, Any],
    accepted: bool,
    is_global_best: bool,
    pdb_filename: str,
    plddt_filename: Optional[str] = None,
    coordinate_shape: Tuple[int, ...] = (),
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    写入 step 级别的元数据 JSON。
    """
    if isinstance(target_seq, str):
        n_target_res = len(target_seq)
    elif isinstance(target_seq, (list, tuple)) and all(isinstance(x, str) for x in target_seq):
        n_target_res = sum(len(x) for x in target_seq)
    else:
        # Best-effort fallback; keep metadata writing robust.
        n_target_res = None

    payload = {
        "step": step,
        "beam_id": beam_id,
        "binder_seq": binder_seq,
        "target_seq": target_seq,
        "total_loss": float(total_loss),
        "loss_dict": {k: float(v) for k, v in loss_dict.items()},
        "summary_confidence": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                               for k, v in summary_confidence.items()},
        "accepted": accepted,
        "is_global_best": is_global_best,
        "pdb_file": pdb_filename,
        "plddt_file": plddt_filename,
        "coordinate_shape": list(coordinate_shape),
        "n_binder_res": len(binder_seq),
        "n_target_res": n_target_res,
    }
    if extra is not None:
        payload["extra"] = extra

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_plddt_json(
    output_path: str,
    step: int,
    beam_id: int,
    plddt_per_atom: np.ndarray,
    chain_boundaries: Dict[str, Tuple[int, int]],
) -> None:
    """
    写入原子级 pLDDT JSON，用于可视化着色。
    """
    payload = {
        "step": step,
        "beam_id": beam_id,
        "plddt_per_atom": plddt_per_atom.tolist() if hasattr(plddt_per_atom, "tolist") else list(plddt_per_atom),
        "chain_boundaries": {k: list(v) for k, v in chain_boundaries.items()},
        "n_atom": len(plddt_per_atom),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


# =============================================================================
# pLDDT logits → 原子级 pLDDT 分数
# =============================================================================

def compute_atom_plddt(
    plddt_logits: torch.Tensor,   # shape [N_atom, bins] 或 [N_atom,]
    min_bin: float = 0.0,
    max_bin: float = 100.0,
    no_bins: int = 50,
) -> np.ndarray:
    """
    将 pLDDT logits 转换为原子级 pLDDT 分数（0~100）。

    如果输入已经是标量分数，直接返回 numpy 数组。
    """
    if plddt_logits.ndim == 1 or (plddt_logits.ndim == 2 and plddt_logits.shape[-1] == 1):
        return _to_numpy(plddt_logits).flatten().astype(np.float64)

    logits = _to_numpy(plddt_logits)
    if logits.ndim == 2:
        # softmax over last axis (bins)
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        prob = e / e.sum(axis=-1, keepdims=True)
        bin_centers = np.linspace(min_bin, max_bin, no_bins)
        plddt = prob @ bin_centers
    else:
        plddt = logits.flatten()
    return plddt.astype(np.float64)


# =============================================================================
# 链边界（用于 plddt JSON 中的 chain_boundaries）
# =============================================================================

def get_chain_boundaries(
    binder_seq: str,
    target_seq: Any,
    atom_array,
) -> Dict[str, Tuple[int, int]]:
    """
    根据 atom_array 的 chain_id 确定 binder / target 的原子范围。

    Returns:
        {"binder": (start, end), "target": (start, end)}
        区间为左闭右开 [start, end)
    """
    chain_ids = np.asarray(atom_array.chain_id, dtype=str)
    binder_start = 0
    binder_end   = int(np.sum(chain_ids == CHAIN_BINDER_STR))
    target_start = binder_end
    target_end   = atom_array.array_length()

    return {
        "binder": (int(binder_start), int(binder_end)),
        "target": (int(target_start), int(target_end)),
    }


# =============================================================================
# CheckpointManager（主类）
# =============================================================================

class CheckpointManager:
    """
    统一管理训练过程的所有 checkpoint 写入。

    每个 step / 每个 beam 保存一次完整复合物结构 + 元数据 + pLDDT。

    使用示例：
        manager = CheckpointManager(output_dir="outputs/checkpoints/8yat")
        manager.save(
            step=0,
            beam_id=0,
            zeroth_result=round_best,
            plddt_logits=plddt_logits_for_best,
            input_feature_dict=input_feature_dict,   # 可选
            accepted=True,
            is_global_best=True,
        )
    """

    def __init__(
        self,
        output_dir: str,
        binder_seq: str = "",
        target_seq: str = "",
        overwrite: bool = True,
    ):
        """
        Args:
            output_dir:   顶级输出目录，如 "outputs/checkpoints/8yat"
            binder_seq:    Binder 序列（提前传入避免重复构建）
            target_seq:    Target 序列
            overwrite:     是否覆盖已有文件
        """
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.binder_seq = binder_seq
        self.target_seq = target_seq

        self._best_loss: float = float("inf")
        self._best_step: int = -1
        self._best_beam: int = -1
        self._best_pdb: Optional[Path] = None
        self._best_json: Optional[Path] = None
        self._best_plddt: Optional[Path] = None

        # 预构建 atom_array（用占位序列，第一次 save 时替换）
        self._atom_array = None
        self._atom_array_cache_key: Optional[str] = None

        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _get_atom_array(self, binder_seq: str, target_seq: str):
        """缓存 atom_array，同一复合物不重复构建。"""
        cache_key = f"{binder_seq}|{target_seq}"
        if self._atom_array is None or self._atom_array_cache_key != cache_key:
            self._atom_array = _rebuild_atom_array(binder_seq, target_seq)
            self._atom_array_cache_key = cache_key
        return self._atom_array

    def _format_filename(self, step: int, beam_id: int, suffix: str, ext: str) -> str:
        return f"step_{step:04d}_beam_{beam_id}_{suffix}.{ext}"

    def _step_dir(self, step: int) -> Path:
        """每个 step 一个子目录（可选，方便查看）。"""
        d = self.output_dir / f"step_{step:04d}"
        d.mkdir(exist_ok=True)
        return d

    def _write_pdb(
        self,
        step: int,
        beam_id: int,
        coordinates: np.ndarray,
        atom_array,
        plddt_per_atom: Optional[np.ndarray] = None,
    ) -> str:
        """写 PDB 并返回文件名。"""
        filename = self._format_filename(step, beam_id, "complex", "pdb")
        out_path = self._step_dir(step) / filename
        write_pdb(
            output_path=str(out_path),
            atom_array=atom_array,
            coordinates=coordinates,
            plddt_per_atom=plddt_per_atom,
        )
        return filename

    def _write_metadata(
        self,
        step: int,
        beam_id: int,
        binder_seq: str,
        target_seq: str,
        total_loss: float,
        loss_dict: Dict[str, float],
        summary_confidence: Dict[str, Any],
        accepted: bool,
        is_global_best: bool,
        pdb_filename: str,
        plddt_filename: Optional[str],
        coordinate_shape: Tuple[int, ...],
    ) -> str:
        """写 JSON 元数据并返回文件名。"""
        filename = self._format_filename(step, beam_id, "meta", "json")
        out_path = self._step_dir(step) / filename
        write_metadata_json(
            output_path=str(out_path),
            step=step,
            beam_id=beam_id,
            binder_seq=binder_seq,
            target_seq=target_seq,
            total_loss=total_loss,
            loss_dict=loss_dict,
            summary_confidence=summary_confidence,
            accepted=accepted,
            is_global_best=is_global_best,
            pdb_filename=pdb_filename,
            plddt_filename=plddt_filename,
            coordinate_shape=coordinate_shape,
        )
        return filename

    def _write_plddt(
        self,
        step: int,
        beam_id: int,
        plddt_per_atom: np.ndarray,
        atom_array,
        binder_seq: str,
        target_seq: str,
    ) -> str:
        """写原子级 pLDDT JSON 并返回文件名。"""
        filename = self._format_filename(step, beam_id, "plddt", "json")
        out_path = self._step_dir(step) / filename
        boundaries = get_chain_boundaries(binder_seq, target_seq, atom_array)
        write_plddt_json(
            output_path=str(out_path),
            step=step,
            beam_id=beam_id,
            plddt_per_atom=plddt_per_atom,
            chain_boundaries=boundaries,
        )
        return filename

    def _update_best(
        self,
        step: int,
        beam_id: int,
        loss: float,
        pdb_path: Path,
        json_path: Path,
        plddt_path: Path,
    ):
        """更新全局最优帧的软链接。"""
        if loss < self._best_loss:
            self._best_loss = loss
            self._best_step = step
            self._best_beam = beam_id
            self._best_pdb = pdb_path
            self._best_json = json_path
            self._best_plddt = plddt_path

            # 更新 best 软链接（强制覆盖）
            for name in ("best.pdb", "best.json", "best_plddt.json"):
                link = self.output_dir / name
                if link.is_symlink() or link.exists():
                    link.unlink()
            if pdb_path.exists():
                (self.output_dir / "best.pdb").symlink_to(pdb_path.relative_to(self.output_dir))
            if json_path.exists():
                (self.output_dir / "best.json").symlink_to(json_path.relative_to(self.output_dir))
            if plddt_path is not None and plddt_path.exists():
                (self.output_dir / "best_plddt.json").symlink_to(plddt_path.relative_to(self.output_dir))

    # ------------------------------------------------------------------
    # 主保存入口
    # ------------------------------------------------------------------

    def save(
        self,
        step: int,
        beam_id: int,
        zeroth_result,               # ZerothOrderResult
        plddt_logits: Optional[torch.Tensor] = None,   # [N_atom, bins] 或 None
        accepted: bool = False,
        is_global_best: bool = False,
    ) -> Dict[str, str]:
        """
        保存单个 step + beam 的完整 checkpoint。

        Args:
            step:             当前优化步数
            beam_id:          当前 beam 编号
            zeroth_result:    ZerothOrderResult，必须包含
                               binder_seq / target_seq / total_loss /
                               loss_dict / summary_confidence / coordinate
            plddt_logits:     可选，pLDDT logits 来自 confidence head
                               形状 [N_atom, bins]；如果不传则跳过 plddt JSON
            accepted:         该帧是否被当前 step 接受
            is_global_best:   该帧是否是目前全局最优

        Returns:
            {"pdb": str, "json": str, "plddt": str}  保存的文件名
        """
        # ── 从 zeroth_result 提取数据 ────────────────────────────────
        binder_seq      = zeroth_result.binder_seq
        target_seq      = zeroth_result.target_seq
        total_loss      = zeroth_result.total_loss
        loss_dict       = zeroth_result.loss_dict
        summary_conf    = zeroth_result.summary_confidence

        # coordinate 形状 [N_sample, N_atom, 3]，取第一个 sample
        coord_tensor    = zeroth_result.coordinate
        if coord_tensor is None:
            raise ValueError(
                f"ZerothOrderResult.coordinate is None for step={step}, beam={beam_id}"
            )
        coordinates     = _to_numpy(coord_tensor[0])   # [N_atom, 3]

        # ── 重建 atom_array ──────────────────────────────────────────
        atom_array      = self._get_atom_array(binder_seq, target_seq)

        # ── pLDDT per atom ────────────────────────────────────────────
        plddt_filename  = None
        plddt_per_atom   = None
        if plddt_logits is not None:
            plddt_per_atom = compute_atom_plddt(plddt_logits)
            plddt_filename = self._format_filename(step, beam_id, "plddt", "json")

        # ── 写 PDB ────────────────────────────────────────────────────
        pdb_filename = self._write_pdb(
            step=step, beam_id=beam_id,
            coordinates=coordinates,
            atom_array=atom_array,
            plddt_per_atom=plddt_per_atom,
        )
        pdb_path = self._step_dir(step) / pdb_filename

        # ── 写 pLDDT JSON ──────────────────────────────────────────────
        if plddt_filename and plddt_per_atom is not None:
            self._write_plddt(
                step=step, beam_id=beam_id,
                plddt_per_atom=plddt_per_atom,
                atom_array=atom_array,
                binder_seq=binder_seq, target_seq=target_seq,
            )

        # ── 写元数据 JSON ─────────────────────────────────────────────
        json_filename = self._write_metadata(
            step=step, beam_id=beam_id,
            binder_seq=binder_seq, target_seq=target_seq,
            total_loss=total_loss, loss_dict=loss_dict,
            summary_confidence=summary_conf,
            accepted=accepted,
            is_global_best=is_global_best,
            pdb_filename=pdb_filename,
            plddt_filename=plddt_filename,
            coordinate_shape=list(coordinates.shape),
        )
        json_path = self._step_dir(step) / json_filename

        # ── 更新全局最优 ──────────────────────────────────────────────
        plddt_path = self._step_dir(step) / plddt_filename if plddt_filename else None
        if accepted or is_global_best:
            self._update_best(
                step=step, beam_id=beam_id,
                loss=total_loss,
                pdb_path=pdb_path,
                json_path=json_path,
                plddt_path=plddt_path,
            )

        return {
            "pdb":   pdb_filename,
            "json":  json_filename,
            "plddt": plddt_filename,
        }

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def get_best_info(self) -> Dict[str, Any]:
        """返回全局最优帧的信息。"""
        return {
            "step":     self._best_step,
            "beam":     self._best_beam,
            "loss":     self._best_loss,
            "pdb":      str(self._best_pdb) if self._best_pdb else None,
            "json":     str(self._best_json) if self._best_json else None,
            "plddt":    str(self._best_plddt) if self._best_plddt else None,
        }

    @property
    def best_loss(self) -> float:
        return self._best_loss

    @property
    def best_step(self) -> int:
        return self._best_step
