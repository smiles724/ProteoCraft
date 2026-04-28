"""
BindCraft v2 - 基础工具函数

提供通用的辅助函数，供其他模块使用
"""

import os
import tempfile

import torch
import numpy as np
from typing import Any, Dict, Tuple, List, Mapping, Optional, Union


# ============================================================
# 氨基酸相关常量
# ============================================================

PROTEIN_RESIDUE_NAMES = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]

AA_1LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

NUM_AMINO_ACIDS = 20
TargetSeqInput = Union[str, List[str]]


def _default_protenix_common_dir() -> Optional[str]:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.environ.get("PROTENIX_COMMON_DIR"),
        os.path.join(project_root, "benchmark", "protenix_runtime", "common"),
        os.path.join(os.environ.get("PROTENIX_ROOT_DIR", ""), "common"),
    ]
    for path in candidates:
        if path and os.path.exists(os.path.join(path, "components.cif")):
            return path
    return None


def configure_protenix_common_paths(common_dir: Optional[str] = None) -> None:
    """
    将 Protenix CCD cache 指向可用的 common 目录。

    Protenix 默认从 ``$PROTENIX_ROOT_DIR/common`` 或 ``~/common`` 读取
    ``components.cif``；BindCraft 仓库的 benchmark runtime 里也带有一份
    common symlink，因此这里在构建 features 前自动修正路径。
    """
    common_dir = common_dir or _default_protenix_common_dir()
    if common_dir is None:
        return

    components_file = os.path.join(common_dir, "components.cif")
    rdkit_mol_file = os.path.join(common_dir, "components.cif.rdkit_mol.pkl")

    from configs.configs_data import data_configs
    from protenix.data.core import ccd

    data_configs["ccd_components_file"] = components_file
    data_configs["ccd_components_rdkit_mol_file"] = rdkit_mol_file
    ccd.COMPONENTS_FILE = components_file
    ccd.RKDIT_MOL_PKL = ccd.Path(rdkit_mol_file)
    ccd.biotite_load_ccd_cif.cache_clear()


# ============================================================
# 序列转换函数
# ============================================================

def sequence_to_one_letter(seq_3letter: List[str]) -> str:
    """
    将三字母氨基酸序列转换为单字母序列
    
    Args:
        seq_3letter: 三字母序列列表，如 ["ALA", "GLY", "CYS"]
        
    Returns:
        单字母序列字符串，如 "AGC"
    """
    return "".join([AA_1LETTER[aa] for aa in seq_3letter])


def one_letter_to_sequence(seq_1letter: str) -> List[str]:
    """
    将单字母序列转换为三字母氨基酸序列
    
    Args:
        seq_1letter: 单字母序列，如 "AGC"
        
    Returns:
        三字母序列列表，如 ["ALA", "GLY", "CYS"]
    """
    reverse_map = {v: k for k, v in AA_1LETTER.items()}
    return [reverse_map[aa] for aa in seq_1letter]


# ============================================================
# 特征构建函数（调用 Protenix）
# ============================================================

def _normalize_chain_sequences(seq_input: TargetSeqInput) -> List[str]:
    """
    将单链或多链输入统一规范化为序列列表。

    支持:
    - 单条序列字符串
    - 逗号分隔的多条序列字符串
    - 序列列表
    """
    if isinstance(seq_input, str):
        parts = [part.strip().upper() for part in seq_input.split(",")]
        seqs = [part for part in parts if part]
    else:
        seqs = [str(seq).strip().upper() for seq in seq_input if str(seq).strip()]

    if not seqs:
        raise ValueError("至少需要提供一条蛋白质序列")

    return seqs


def _convert_pdb_to_cif_string(template_path: str, target_seqs: List[str]) -> str:
    from protenix.data.utils import pdb_to_cif

    suffix = os.path.splitext(template_path)[1].lower()
    if suffix in {".cif", ".mmcif"}:
        with open(template_path, "r") as f:
            return f.read()
    if suffix != ".pdb":
        raise ValueError(
            f"target template 仅支持 .cif/.mmcif/.pdb，收到: {template_path}"
        )

    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        cif_path = tmp.name
    try:
        pdb_to_cif(template_path, cif_path)
        with open(cif_path, "r") as f:
            return f.read()
    finally:
        try:
            os.remove(cif_path)
        except OSError:
            pass


def _align_score(query_seq: str, template_seq: str) -> Tuple[float, int, int]:
    from difflib import SequenceMatcher

    matcher = SequenceMatcher(None, query_seq, template_seq, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    coverage = matched / max(len(query_seq), 1)
    identity = matched / max(min(len(query_seq), len(template_seq)), 1)
    return coverage + identity, matched, abs(len(query_seq) - len(template_seq))


def _match_target_template_chains(
    target_seqs: List[str],
    chain_to_seq: Mapping[str, str],
) -> List[str]:
    available = set(chain_to_seq.keys())
    matched_chains = []

    for target_idx, target_seq in enumerate(target_seqs):
        candidates = []
        for chain_id in available:
            score, matched, length_delta = _align_score(target_seq, chain_to_seq[chain_id])
            candidates.append((score, matched, -length_delta, chain_id))

        if not candidates:
            raise ValueError(
                f"target template 链数量不足，无法匹配第 {target_idx + 1} 条 target 链"
            )

        score, matched, _, chain_id = max(candidates)
        coverage = matched / max(len(target_seq), 1)
        if coverage < 0.5:
            raise ValueError(
                f"target template 中没有足够匹配第 {target_idx + 1} 条 target 链的结构链 "
                f"(best_chain={chain_id}, coverage={coverage:.2f})"
            )
        matched_chains.append(chain_id)
        available.remove(chain_id)

    return matched_chains


def _map_query_to_template_indices(query_seq: str, template_seq: str) -> Dict[int, int]:
    if query_seq == template_seq:
        return {idx: idx for idx in range(len(query_seq))}

    start = template_seq.find(query_seq)
    if start >= 0:
        return {idx: start + idx for idx in range(len(query_seq))}

    start = query_seq.find(template_seq)
    if start >= 0:
        return {start + idx: idx for idx in range(len(template_seq))}

    return {}


def _normalize_template_release_date(release_date: str) -> str:
    if not release_date or release_date in {".", "?"}:
        return "1970-01-01"
    return release_date


def _extract_target_template_features(
    target_seqs: List[str],
    target_template_path: str,
) -> Dict[str, Any]:
    from protenix.data.template.template_parser import TemplateParser
    from protenix.data.template.template_utils import TemplateHitProcessor

    if not os.path.exists(target_template_path):
        raise FileNotFoundError(f"target template 文件不存在: {target_template_path}")

    cif_string = _convert_pdb_to_cif_string(target_template_path, target_seqs)
    file_id = os.path.splitext(os.path.basename(target_template_path))[0]
    parsed = TemplateParser.parse(file_id=file_id, mmcif_string=cif_string)
    if parsed.mmcif_object is None:
        raise ValueError(f"target template 解析失败: {target_template_path}; errors={parsed.errors}")

    matched_chains = _match_target_template_chains(
        target_seqs, parsed.mmcif_object.chain_to_seqres
    )
    processor = TemplateHitProcessor(mmcif_dir="", _zero_center_positions=True)

    templates_by_target_idx = {}
    for target_idx, (query_seq, chain_id) in enumerate(zip(target_seqs, matched_chains)):
        actual_chain_id = chain_id
        template_seq = parsed.mmcif_object.chain_to_seqres[chain_id]
        mapping = _map_query_to_template_indices(query_seq, template_seq)
        if not mapping:
            template_seq, mapping, actual_chain_id = processor._align_query_to_hit_index_mapping(
                query_seq=query_seq,
                chain_id=chain_id,
                mmcif_obj=parsed.mmcif_object,
                min_ratio=0.0,
            )
        features, _ = processor._extract_template_features(
            mmcif_obj=parsed.mmcif_object,
            pdb_id=file_id,
            mapping=mapping,
            template_seq=template_seq,
            query_seq=query_seq,
            chain_id=actual_chain_id,
            _zero_center=True,
        )
        features["template_sum_probs"] = [1.0]
        release_date = _normalize_template_release_date(
            parsed.mmcif_object.header.get("release_date", "1970-01-01")
        )
        features["template_release_date"] = np.array(release_date.encode(), dtype=object)
        templates_by_target_idx[target_idx] = [features]

    return templates_by_target_idx


def _make_template_features_from_structure(
    sample_dict: Dict[str, Any],
    atom_array,
    target_template_path: Optional[str],
) -> Dict[str, np.ndarray]:
    from protenix.data.constants import PROTEIN_CHAIN
    from protenix.data.msa.msa_utils import map_to_standard
    from protenix.data.template.template_featurizer import TemplateFeatureAssemblyLine

    target_seqs = [
        entity["proteinChain"]["sequence"]
        for entity in sample_dict["sequences"][1:]
    ]
    templates_by_target_idx = (
        _extract_target_template_features(target_seqs, target_template_path)
        if target_template_path
        else {}
    )

    template_meta_infos = {}
    curr_asym_id = 0
    for entity_idx, entity in enumerate(sample_dict["sequences"]):
        chain = entity["proteinChain"]
        count = chain["count"]
        target_idx = entity_idx - 1
        templates = templates_by_target_idx.get(target_idx, [])
        for i in range(count):
            asym_id = curr_asym_id + i
            template_meta_infos[asym_id] = {
                "entity_id": asym_id,
                "chain_id": atom_array.chain_id[atom_array.asym_id_int == asym_id][0],
                "sequence": chain["sequence"],
                "chain_entity_type": PROTEIN_CHAIN,
                "templates": templates,
            }
        curr_asym_id += count

    ca = atom_array[atom_array.centre_atom_mask.astype(bool)]
    standard_idxs = map_to_standard(ca.asym_id_int, ca.res_id, template_meta_infos)
    return (
        TemplateFeatureAssemblyLine(max_templates=4)
        .assemble(template_meta_infos, standard_idxs)
        .as_protenix_dict()
    )


def _build_features_from_sequences(
    seq_input: TargetSeqInput,
    device: torch.device,
    name: str,
    target_template_path: Optional[str] = None,
) -> Tuple[Dict, int, int]:
    """
    基于一条或多条蛋白链序列构建 Protenix 输入特征。
    """
    configure_protenix_common_paths()

    from protenix.data.inference.json_to_feature import SampleDictToFeatures
    from protenix.data.inference.infer_dataloader import InferenceTemplateFeaturizer
    from protenix.data.utils import data_type_transform, make_dummy_feature
    from protenix.utils.torch_utils import dict_to_tensor

    seqs = _normalize_chain_sequences(seq_input)
    sample_dict = {
        "sequences": [
            {"proteinChain": {"sequence": seq, "count": 1}}
            for seq in seqs
        ],
        "name": name,
    }

    sample2feat = SampleDictToFeatures(sample_dict)
    features_dict, atom_array, token_array = sample2feat.get_feature_dict()
    features_dict["distogram_rep_atom_mask"] = torch.Tensor(
        atom_array.distogram_rep_atom_mask
    ).long()

    if target_template_path:
        template_features = _make_template_features_from_structure(
            sample_dict=sample_dict,
            atom_array=atom_array,
            target_template_path=target_template_path,
        )
    else:
        template_features = InferenceTemplateFeaturizer.make_template_feature(
            bioassembly=sample_dict["sequences"],
            atom_array=atom_array,
            use_template=False,
            online_template_featurizer=None,
        )

    dummy_feats = ["msa"]
    if len(template_features) == 0:
        dummy_feats.append("template")
    else:
        features_dict.update(dict_to_tensor(template_features))
    features_dict = make_dummy_feature(features_dict=features_dict, dummy_feats=dummy_feats)

    feat = data_type_transform(feat_or_label_dict=features_dict)

    input_feature_dict = {}
    for k, v in feat.items():
        if isinstance(v, torch.Tensor):
            input_feature_dict[k] = v.to(device)
        else:
            input_feature_dict[k] = v

    N_token = feat["token_index"].shape[0]
    N_atom = feat["atom_to_token_idx"].shape[0]

    return input_feature_dict, N_token, N_atom


def build_single_chain_features(seq: TargetSeqInput, device: torch.device) -> Tuple[Dict, int, int]:
    """
    构建蛋白质目标的输入特征（支持单链或多链）。

    当输入为单链时，可用于 W_AAE 预计算。
    当输入为多链时，可用于提取多链 target-only 特征。

    Args:
        seq: 单字母蛋白质序列，或多条序列组成的列表/逗号分隔字符串
        device: 计算设备

    Returns:
        input_feature_dict: 输入特征字典
        N_token: Token 总数
        N_atom: 原子总数
    """
    return _build_features_from_sequences(seq, device, name="target_only")


def build_complex_features(
    binder_seq: str,
    target_seq: TargetSeqInput,
    device: torch.device,
    target_template_path: Optional[str] = None,
) -> Tuple[Dict, int, int]:
    """
    构建 Binder-Target 复合物的输入特征，支持单链或多链 target。
    
    Args:
        binder_seq: Binder 单字母序列
        target_seq: Target 单字母序列，或多条 target 链组成的列表/逗号分隔字符串
        device: 计算设备
        target_template_path: 可选 target template 结构文件（.cif/.mmcif/.pdb）
        
    Returns:
        input_feature_dict: 输入特征字典
        N_token: Token 总数
        N_atom: 原子总数
    """
    target_seqs = _normalize_chain_sequences(target_seq)
    seqs = [binder_seq] + target_seqs
    return _build_features_from_sequences(
        seqs,
        device,
        name="binder_target_complex",
        target_template_path=target_template_path,
    )


def prepare_input_features(model, input_feature_dict: Dict) -> Dict:
    """
    准备输入特征（添加相对位置编码等）
    
    Args:
        model: Protenix 模型
        input_feature_dict: 输入特征字典
        
    Returns:
        更新后的特征字典
    """
    from protenix.model.protenix import update_input_feature_dict
    
    # 生成相对位置编码
    input_feature_dict = model.relative_position_encoding.generate_relp(input_feature_dict)
    
    # 更新特征字典
    input_feature_dict = update_input_feature_dict(input_feature_dict)
    
    return input_feature_dict


# ============================================================
# 模型冻结函数
# ============================================================

def freeze_model(model):
    """
    冻结模型的所有参数
    
    Args:
        model: Protenix 模型
    """
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


# ============================================================
# 日志工具
# ============================================================

def log_step(step: int, total_steps: int, losses: Dict[str, float], 
             sequence: str, extra_info: str = ""):
    """
    打印优化步骤的日志
    
    Args:
        step: 当前步数
        total_steps: 总步数
        losses: 损失字典
        sequence: 当前序列
        extra_info: 额外信息
    """
    loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
    print(f"Step {step:3d}/{total_steps} | {loss_str} | seq: {sequence[:20]}... {extra_info}")


# ============================================================
# 默认 Loss 权重与 Contact 参数
# （对标原版 BindCraft default_4stage_multimer.json）
# 放在 utils 中供 pipeline / zeroth_order 等多个模块共享，
# 避免循环导入。
# ============================================================

DEFAULT_LOSS_WEIGHTS = {
    "plddt":   0.1,
    "pae":     0.4,
    "i_pae":   0.1,
    "con":     1.0,
    "i_con":   3.0,
    "iptm":    0.05,
    "rg":      0.3,
    "helix":  -0.3,
    "clash":   0.5,
    "i_clash": 0.5,
    "pde":     0.02,
    "i_pde":   0.01,
}

DEFAULT_CONTACT_PARAMS = {
    "intra_cutoff":  14.0,
    "inter_cutoff":  21.6875,
    "intra_num":     2,
    "inter_num":     1,
    "inter_num_pos": 100000,  # v1: num_pos=inf，取所有位置
    "seqsep":        9,
}

DEFAULT_CLASH_PARAMS = {
    "r_clash":   2.0,
    "bond_sep":  1,
}
