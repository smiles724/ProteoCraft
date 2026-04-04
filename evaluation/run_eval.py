#!/usr/bin/env python3
"""
BindCraft v2 — 统一评估入口

支持两种评估模式：

  1. zeroth (默认, 快速):
     读取 evaluation/outputs/designs/*.json，对 designed/real_vh/real_full
     三组序列调用 zeroth_order_eval()（完整ODE采样 + 8项loss计算）。
     输出: evaluation/outputs/eval_results.csv

  2. full (深度, 高精度):
     对任意序列执行 Protenix 多步 ODE 扩散采样（可多 seed），提取原生置信度指标。
     输出: evaluation/outputs/full_eval_results.csv

  3. all: 依次运行 zeroth + full。

用法（zeroth 模式 — 快速批量评估）:
    CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py --eval_mode zeroth

用法（full 模式 — 高精度深度评估）:
    CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py --eval_mode full

用法（all 模式 — 同时运行两种评估）:
    CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py --eval_mode all

单 case 快速测试（full 模式）:
    CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py --eval_mode full \
        --binder_seq QVQLVESGGGLVQAGGSLRLSCAASGRTFSRYAMIWFRQAPGKEREFVSALSISIDPTDYADSVKGRFTISRDNAKSTVYLRMSSLKPEDTAVYYCAAEGSRLSSTRLDARFTYDYWGQGTQVTVSS \
        --target_seq TLWCGFKHTDACCRTHDMCPDVMSAGESKHGLTNTASHTRLSCDCDDKFYDCLKNSADTISSYFVGKMYFNLIDTKCYKLEHPVTGCGERTEGRCLHYTVDKSKPKVYQWFDLRKY \
        --label 8PIH_DesignedBest
"""

import os
import sys
import csv
import json
import glob
import time
import random
import logging
import argparse
from datetime import datetime
from typing import Optional

import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.configs import (
    OUTPUT_DIR, DESIGNS_DIR, LOG_DIR, MODEL_NAME,
    AB_RAW_CSV, AB_RAW_MODEL_PREFIX, EVAL_MAX_TOTAL_SEQ_LEN,
    EVAL_N_REPEATS_FULL, EVAL_N_CYCLES,
)

# ─── 日志配置 ─────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)
log_filepath = os.path.join(
    LOG_DIR, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filepath, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# 模式 1: zeroth_order_eval（快速批量评估）
# ═══════════════════════════════════════════════════════════════════

ZEROTH_RESULT_COLUMNS = [
    "entry_id", "seq_type", "binder_length", "target_length",
    "total_loss",
    # ── 零阶优化目标（与 pipeline loss 对齐）────────────────
    "plddt", "pae", "i_pae", "con", "i_con", "iptm", "rg", "helix",
    "clash", "i_clash", "pde", "i_pde",
    # ── Protenix 原生置信度（独立于优化目标）────────────────
    "mean_plddt_conf", "iptm_conf", "ptm_conf",
    "dockq_ref", "lddt_ref",
    "eval_time_sec",
]


def load_dockq_reference(ab_raw_path: str, model_prefix: str) -> dict:
    """
    从 ab-raw.csv 读取每个 entry_id 在指定模型下的 best-DockQ，
    仅作为参考列附在 real 组结果中。
    """
    best = {}
    with open(ab_raw_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if model_prefix not in row.get("model_name", ""):
                continue
            eid = row["entry_id"]
            try:
                dq = float(row["dockq"])
            except (ValueError, KeyError):
                continue
            if eid not in best or dq > best[eid]["dockq"]:
                best[eid] = {"dockq": dq, "lddt": float(row.get("lddt", 0.0))}
    return best


def eval_one_sequence_zeroth(
    binder_seq: str,
    target_seq: str,
    model,
    device: torch.device,
):
    """调用 zeroth_order_eval，返回扁平化的指标字典。"""
    from bindcraft.hallucination.zeroth_order import zeroth_order_eval

    t0 = time.time()
    result = zeroth_order_eval(binder_seq=binder_seq, target_seq=target_seq,
                               model=model, device=device)
    elapsed = time.time() - t0

    row = {
        "total_loss": result.total_loss,
        "eval_time_sec": round(elapsed, 1),
    }
    for key in ["plddt", "pae", "i_pae", "con", "i_con", "iptm", "rg",
                "helix", "clash", "i_clash", "pde", "i_pde"]:
        row[key] = result.loss_dict.get(key, 0.0)

    sc = result.summary_confidence
    row["mean_plddt_conf"] = sc.get("plddt", sc.get("mean_plddt", 0.0))
    row["iptm_conf"] = sc.get("iptm", 0.0)
    row["ptm_conf"] = sc.get("ptm", 0.0)

    return row


def get_real_binder_for_eval(data: dict, target_len: int, binder_key: str) -> Optional[str]:
    """从 design JSON 中提取指定键的 real binder 序列，超长则返回 None。"""
    binder = data.get(binder_key, "")
    if not binder:
        return None
    if target_len + len(binder) > EVAL_MAX_TOTAL_SEQ_LEN:
        return None
    return binder


def run_zeroth_evaluation(device_str: str, entry_ids: list = None):
    """
    zeroth_order_eval 模式：从 designs/*.json 读取，对四组序列快速评估。
    输出到 evaluation/outputs/eval_results.csv。
    """
    from scripts.run_hallucination import load_protenix_model

    design_files = sorted(glob.glob(os.path.join(DESIGNS_DIR, "*.json")))
    if not design_files:
        logger.warning(f"未找到设计结果: {DESIGNS_DIR}/*.json（zeroth 模式跳过）")
        return

    if entry_ids:
        design_files = [f for f in design_files if any(eid in f for eid in entry_ids)]
        logger.info(f"按 --entry_ids 过滤后: {len(design_files)} 个设计结果待评估")

    logger.info(f"[zeroth 模式] 共 {len(design_files)} 个设计结果待评估")
    logger.info(f"加载 ab-raw.csv DockQ 参考 ({AB_RAW_MODEL_PREFIX}) ...")
    dockq_ref = load_dockq_reference(AB_RAW_CSV, AB_RAW_MODEL_PREFIX)
    logger.info(f"  共 {len(dockq_ref)} 个 entry 有 DockQ 参考")

    device = torch.device(device_str)
    logger.info(f"加载模型 {MODEL_NAME} -> {device}")
    model, _ = load_protenix_model(MODEL_NAME, str(device))

    out_path = os.path.join(OUTPUT_DIR, "eval_results.csv")
    file_exists = os.path.exists(out_path)
    mode = "a" if file_exists else "w"
    with open(out_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ZEROTH_RESULT_COLUMNS)
        if not file_exists:
            writer.writeheader()

    total_evals = 0

    for idx, dfile in enumerate(design_files):
        with open(dfile) as f:
            data = json.load(f)

        entry_id = data["entry_id"]
        target_seq = data["target_seq"]
        target_len = len(target_seq)

        sequences = {
            "designed": data["designed_seq"],
        }

        real_vh_binder = get_real_binder_for_eval(data, target_len, "real_binder_heavy")
        if real_vh_binder:
            sequences["real_vh"] = real_vh_binder
        else:
            heavy_raw = data.get("real_binder_heavy", "")
            if heavy_raw:
                logger.warning(
                    f"  {entry_id}: target({target_len}) + VH({len(heavy_raw)}) = "
                    f"{target_len + len(heavy_raw)}aa > {EVAL_MAX_TOTAL_SEQ_LEN}，跳过 real_vh"
                )
            else:
                logger.warning(f"  {entry_id}: 无 real_binder_heavy，跳过 real_vh")

        real_full_binder = get_real_binder_for_eval(data, target_len, "real_binder_seq")
        if real_full_binder:
            sequences["real_full"] = real_full_binder
        else:
            full_raw = data.get("real_binder_seq", "")
            if full_raw:
                logger.warning(
                    f"  {entry_id}: target({target_len}) + H+L({len(full_raw)}) = "
                    f"{target_len + len(full_raw)}aa > {EVAL_MAX_TOTAL_SEQ_LEN}，跳过 real_full"
                )
            else:
                logger.warning(f"  {entry_id}: 无 real_binder_seq，跳过 real_full")

        logger.info("=" * 60)
        logger.info(
            f"[{idx+1}/{len(design_files)}] 评估 {entry_id} | "
            f"target={target_len}aa | {len(sequences)} 组序列"
        )

        gt = dockq_ref.get(entry_id)

        for seq_type, binder_seq in sequences.items():
            logger.info(f"  {seq_type}: {binder_seq[:20]}... ({len(binder_seq)}aa)")
            try:
                metrics = eval_one_sequence_zeroth(binder_seq, target_seq, model, device)
            except Exception as e:
                logger.error(f"  评估失败 ({seq_type}): {e}")
                continue

            row = {
                "entry_id": entry_id,
                "seq_type": seq_type,
                "binder_length": len(binder_seq),
                "target_length": target_len,
                "dockq_ref": "",
                "lddt_ref": "",
                **metrics,
            }
            if seq_type == "real_full" and gt:
                row["dockq_ref"] = gt["dockq"]
                row["lddt_ref"] = gt["lddt"]

            with open(out_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ZEROTH_RESULT_COLUMNS)
                writer.writerow(row)

            total_evals += 1
            extra = f" | DockQ_ref={gt['dockq']:.4f}" if (seq_type == "real_full" and gt) else ""
            logger.info(
                f"    total_loss={metrics['total_loss']:.4f} | "
                f"plddt_conf={metrics['mean_plddt_conf']:.1f} | "
                f"iptm_conf={metrics['iptm_conf']:.3f} | "
                f"ptm_conf={metrics['ptm_conf']:.3f} | "
                f"{metrics['eval_time_sec']}s{extra}"
            )

    logger.info("=" * 60)
    logger.info(
        f"[zeroth 模式] 完成 | 共 {total_evals} 次推断 | "
        f"结果: {out_path} | 日志: {log_filepath}"
    )


# ═══════════════════════════════════════════════════════════════════
# 模式 2: full ODE inference（高精度深度评估）
# ═══════════════════════════════════════════════════════════════════

FULL_RESULT_COLUMNS = [
    "label", "seq_type", "binder_length", "target_length",
    "plddt_mean", "plddt_std", "plddt_min", "plddt_max",
    "iptm_mean", "iptm_std",
    "ptm_mean", "ptm_std",
    "ranking_score_mean", "ranking_score_std",
    "chain_pair_iptm_bt_mean", "chain_pair_iptm_bt_std",
    "chain_pair_iptm_tb_mean", "chain_pair_iptm_tb_std",
    "gpde_mean", "gpde_std",
    "has_clash_frac",
    "eval_time_per_run_sec",
]


def _run_full_inference_once(
    model,
    binder_seq: str,
    target_seq: str,
    device: torch.device,
    seed: int = 0,
    N_cycle: int = 4,
) -> dict:
    """
    单次完整 ODE 推理，返回 summary_confidence 中的所有指标。
    """
    from bindcraft.utils import build_complex_features, prepare_input_features

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        input_feature_dict, _, _ = build_complex_features(binder_seq, target_seq, device)
        input_feature_dict = prepare_input_features(model, input_feature_dict)

        pred_dict, _, _ = model._main_inference_loop(
            input_feature_dict=input_feature_dict,
            label_dict=None,
            N_cycle=N_cycle,
            mode="inference",
            inplace_safe=True,
            chunk_size=None,
        )

    sc_list = pred_dict["summary_confidence"]
    sc = sc_list[0] if isinstance(sc_list, list) else sc_list

    def _s(t):
        if isinstance(t, torch.Tensor):
            return float(t.flatten()[0].item())
        return float(t)

    r = {
        "plddt":        _s(sc.get("plddt", 0.0)),
        "iptm":         _s(sc.get("iptm", 0.0)),
        "ptm":          _s(sc.get("ptm", 0.0)),
        "ranking_score": _s(sc.get("ranking_score", 0.0)),
        "gpde":         _s(sc.get("gpde", 0.0)),
        "has_clash":    int(_s(sc.get("has_clash", 0.0))),
    }

    cpl = sc.get("chain_plddt")
    if isinstance(cpl, torch.Tensor):
        cpl = cpl.flatten()
        if len(cpl) >= 2:
            r["chain_plddt_binder"] = float(cpl[0].item())
            r["chain_plddt_target"] = float(cpl[1].item())
        elif len(cpl) == 1:
            r["chain_plddt_binder"] = float(cpl[0].item())

    cpi = sc.get("chain_pair_iptm")
    if isinstance(cpi, torch.Tensor):
        if cpi.dim() == 2 and cpi.shape[0] >= 2 and cpi.shape[1] >= 2:
            r["chain_pair_iptm_bt"] = float(cpi[0, 1].item())
            r["chain_pair_iptm_tb"] = float(cpi[1, 0].item())

    return r


def evaluate_seq_full_ode(
    model,
    binder_seq: str,
    target_seq: str,
    device: torch.device,
    n_repeats: int = 3,
    N_cycle: int = 4,
    label: str = "eval",
) -> dict:
    """
    重复 n_repeats 次完整 ODE 推理，统计均值±标准差、最小最大值。
    """
    logger.info(
        f"[{label}] 序列: {binder_seq[:30]}... ({len(binder_seq)} aa) | "
        f"{n_repeats}x ODE(N_cycle={N_cycle})"
    )

    all_results = []
    for i in range(n_repeats):
        seed = 42 + i * 111
        t0 = time.time()
        res = _run_full_inference_once(
            model, binder_seq, target_seq, device, seed=seed, N_cycle=N_cycle
        )
        elapsed = time.time() - t0
        all_results.append({**res, "_elapsed": elapsed})

        logger.info(
            f"  [{i+1}/{n_repeats} seed={seed}] "
            f"pLDDT={res['plddt']:.1f}  "
            f"iPTM={res['iptm']:.3f}  "
            f"pTM={res['ptm']:.3f}  "
            f"ranking={res['ranking_score']:.3f}  "
            f"gpde={res['gpde']:.3f}  "
            f"[{elapsed:.1f}s]"
        )

    keys = list(all_results[0].keys())
    keys_stat = [k for k in keys if not k.startswith("_")]
    stats = {}
    for k in keys_stat:
        vals = [r[k] for r in all_results]
        stats[k] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
        }

    avg_time = np.mean([r["_elapsed"] for r in all_results])
    stats["_eval_time_per_run"] = round(avg_time, 1)
    stats["_has_clash_frac"] = round(
        sum(r["has_clash"] for r in all_results) / len(all_results), 2
    )

    logger.info(
        f"  汇总 | "
        f"pLDDT={stats['plddt']['mean']:.1f}±{stats['plddt']['std']:.1f} | "
        f"iPTM={stats['iptm']['mean']:.3f}±{stats['iptm']['std']:.3f} | "
        f"gpde={stats['gpde']['mean']:.3f}±{stats['gpde']['std']:.3f}"
    )
    return stats


def run_full_ode_evaluation(
    device_str: str,
    binder_seq: Optional[str] = None,
    target_seq: Optional[str] = None,
    label: Optional[str] = None,
    design_entry_id: Optional[str] = None,
    entry_ids: list = None,
    n_repeats: int = None,
    N_cycle: int = None,
):
    """
    full ODE 模式：可对单序列、自定义序列列表、或 designs/*.json 中的案例进行评估。

    支持三种输入组合：
      (a) --binder_seq + --target_seq + --label:  单序列评估
      (b) --design_entry_id:                       从 designs/ 读取指定 entry
      (c) 无特殊参数:                              评估所有 designs/*.json

    输出到 evaluation/outputs/full_eval_results.csv。
    """
    from scripts.run_hallucination import load_protenix_model

    n_repeats = n_repeats or EVAL_N_REPEATS_FULL
    N_cycle   = N_cycle   or EVAL_N_CYCLES

    device = torch.device(device_str)
    logger.info(f"[full ODE 模式] 加载模型 {MODEL_NAME} -> {device}")
    model, _ = load_protenix_model(MODEL_NAME, str(device))

    out_path = os.path.join(OUTPUT_DIR, "full_eval_results.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_exists = os.path.exists(out_path)
    mode = "a" if file_exists else "w"

    # 确保所有可能的列都在 fieldnames 中
    fieldnames = list(FULL_RESULT_COLUMNS)
    for extra in ["chain_plddt_binder", "chain_plddt_target"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    with open(out_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    # ── 模式 (a): 单序列评估 ─────────────────────────────────────
    if binder_seq and target_seq:
        seq_label = label or f"custom_{len(binder_seq)}aa"
        logger.info("=" * 70)
        logger.info(f"[full ODE] 单序列评估: {seq_label}")
        logger.info("=" * 70)

        stats = evaluate_seq_full_ode(
            model, binder_seq, target_seq, device,
            n_repeats=n_repeats, N_cycle=N_cycle, label=seq_label
        )

        row = {
            "label": seq_label,
            "seq_type": "custom",
            "binder_length": len(binder_seq),
            "target_length": len(target_seq),
            "plddt_mean":      round(stats["plddt"]["mean"], 4),
            "plddt_std":       round(stats["plddt"]["std"], 4),
            "plddt_min":       round(stats["plddt"]["min"], 4),
            "plddt_max":       round(stats["plddt"]["max"], 4),
            "iptm_mean":       round(stats["iptm"]["mean"], 4),
            "iptm_std":        round(stats["iptm"]["std"], 4),
            "ptm_mean":        round(stats["ptm"]["mean"], 4),
            "ptm_std":         round(stats["ptm"]["std"], 4),
            "ranking_score_mean": round(stats["ranking_score"]["mean"], 4),
            "ranking_score_std":  round(stats["ranking_score"]["std"], 4),
            "chain_pair_iptm_bt_mean": round(stats.get("chain_pair_iptm_bt", {}).get("mean", 0.0), 4),
            "chain_pair_iptm_bt_std":  round(stats.get("chain_pair_iptm_bt", {}).get("std", 0.0), 4),
            "chain_pair_iptm_tb_mean": round(stats.get("chain_pair_iptm_tb", {}).get("mean", 0.0), 4),
            "chain_pair_iptm_tb_std":  round(stats.get("chain_pair_iptm_tb", {}).get("std", 0.0), 4),
            "gpde_mean":       round(stats["gpde"]["mean"], 4),
            "gpde_std":        round(stats["gpde"]["std"], 4),
            "has_clash_frac":  stats["_has_clash_frac"],
            "eval_time_per_run_sec": stats["_eval_time_per_run"],
        }
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        logger.info(f"  结果已写入: {out_path}")
        return

    # ── 模式 (b/c): 从 designs/*.json 批量评估 ───────────────────
    if design_entry_id:
        design_files = [os.path.join(DESIGNS_DIR, f"{design_entry_id}.json")]
        if not os.path.exists(design_files[0]):
            logger.error(f"未找到设计结果: {design_files[0]}")
            return
    else:
        design_files = sorted(glob.glob(os.path.join(DESIGNS_DIR, "*.json")))
        if not design_files:
            logger.warning(f"未找到设计结果: {DESIGNS_DIR}/*.json（full 模式跳过）")
            return

    if entry_ids:
        design_files = [f for f in design_files if any(eid in f for eid in entry_ids)]
        logger.info(f"按 --entry_ids 过滤后: {len(design_files)} 个设计结果待评估")

    # 定义序列组
    def build_seq_groups(data: dict):
        groups = {}
        target_len = len(data["target_seq"])

        groups["designed"] = data["designed_seq"]

        # real_vh: 仅重链
        real_vh = get_real_binder_for_eval(data, target_len, "real_binder_heavy")
        if real_vh:
            groups["real_vh"] = real_vh

        # real_full: 完整 H+L
        real_full = get_real_binder_for_eval(data, target_len, "real_binder_seq")
        if real_full:
            groups["real_full"] = real_full

        return groups

    total_runs = 0
    for idx, dfile in enumerate(design_files):
        with open(dfile) as f:
            data = json.load(f)

        entry_id = data["entry_id"]
        target_seq_str = data["target_seq"]
        seq_groups = build_seq_groups(data)

        logger.info("=" * 70)
        logger.info(
            f"[{idx+1}/{len(design_files)}] full ODE 评估: {entry_id} | "
            f"target={len(target_seq_str)}aa | {len(seq_groups)} 组"
        )

        for stype, binder_seq_str in seq_groups.items():
            logger.info(f"  {stype}: {binder_seq_str[:20]}... ({len(binder_seq_str)}aa)")

            try:
                stats = evaluate_seq_full_ode(
                    model, binder_seq_str, target_seq_str, device,
                    n_repeats=n_repeats, N_cycle=N_cycle,
                    label=f"{entry_id}_{stype}"
                )
            except Exception as e:
                logger.error(f"  评估失败 ({stype}): {e}")
                continue

            row = {
                "label":                entry_id,
                "seq_type":             stype,
                "binder_length":        len(binder_seq_str),
                "target_length":        len(target_seq_str),
                "plddt_mean":          round(stats["plddt"]["mean"], 4),
                "plddt_std":           round(stats["plddt"]["std"], 4),
                "plddt_min":           round(stats["plddt"]["min"], 4),
                "plddt_max":           round(stats["plddt"]["max"], 4),
                "iptm_mean":           round(stats["iptm"]["mean"], 4),
                "iptm_std":            round(stats["iptm"]["std"], 4),
                "ptm_mean":           round(stats["ptm"]["mean"], 4),
                "ptm_std":            round(stats["ptm"]["std"], 4),
                "ranking_score_mean":  round(stats["ranking_score"]["mean"], 4),
                "ranking_score_std":   round(stats["ranking_score"]["std"], 4),
                "chain_pair_iptm_bt_mean": round(stats.get("chain_pair_iptm_bt", {}).get("mean", 0.0), 4),
                "chain_pair_iptm_bt_std":  round(stats.get("chain_pair_iptm_bt", {}).get("std", 0.0), 4),
                "chain_pair_iptm_tb_mean": round(stats.get("chain_pair_iptm_tb", {}).get("mean", 0.0), 4),
                "chain_pair_iptm_tb_std":  round(stats.get("chain_pair_iptm_tb", {}).get("std", 0.0), 4),
                "gpde_mean":           round(stats["gpde"]["mean"], 4),
                "gpde_std":            round(stats["gpde"]["std"], 4),
                "has_clash_frac":      stats["_has_clash_frac"],
                "eval_time_per_run_sec": stats["_eval_time_per_run"],
            }
            with open(out_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
            total_runs += 1

    logger.info("=" * 70)
    logger.info(
        f"[full ODE 模式] 完成 | 共 {total_runs} 次评估 | "
        f"结果: {out_path} | 日志: {log_filepath}"
    )


# ═══════════════════════════════════════════════════════════════════
# CLI 主入口
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BindCraft v2 — 统一评估入口（支持 zeroth 快速 / full ODE 高精度两种模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--eval_mode", type=str, default="all",
        choices=["zeroth", "full", "all"],
        help="zeroth=快速零阶评估（读 designs/*.json，输出 eval_results.csv）| "
             "full=深度 ODE 评估（可单序列，输出 full_eval_results.csv）| "
             "all=依次运行 zeroth + full"
    )
    parser.add_argument("--device", type=str, default="cuda:0")

    # zeroth 模式专属
    parser.add_argument(
        "--entry_ids", type=str, default=None,
        help="逗号分隔的 entry_id 列表，如 '8trt,9ho4'（仅 zeroth/full 批量模式）"
    )

    # full 模式专属 — 单序列评估
    parser.add_argument("--binder_seq", type=str, default=None,
                        help="Binder 氨基酸序列（full 模式：单序列评估）")
    parser.add_argument("--target_seq", type=str, default=None,
                        help="Target 氨基酸序列（full 模式：单序列评估）")
    parser.add_argument("--label", type=str, default=None,
                        help="评估标签（用于 CSV 中区分不同序列）")
    parser.add_argument(
        "--design_entry_id", type=str, default=None,
        help="从 designs/ 目录读取指定 entry_id 的结果进行 full ODE 评估（full 模式）"
    )

    # full 模式专属 — 精度参数
    parser.add_argument("--n_repeats", type=int, default=None,
                        help=f"每次序列的 ODE 推理次数（默认: {EVAL_N_REPEATS_FULL}）")
    parser.add_argument("--n_cycles", type=int, default=None,
                        help=f"ODE diffusion cycles（默认: {EVAL_N_CYCLES}）")

    args = parser.parse_args()

    entry_ids = None
    if args.entry_ids:
        entry_ids = [e.strip() for e in args.entry_ids.split(",") if e.strip()]

    if args.eval_mode == "zeroth":
        logger.info("=" * 70)
        logger.info("模式: zeroth（零阶快速评估）")
        logger.info("=" * 70)
        run_zeroth_evaluation(args.device, entry_ids)

    elif args.eval_mode == "full":
        logger.info("=" * 70)
        logger.info("模式: full ODE（高精度深度评估）")
        logger.info("=" * 70)
        run_full_ode_evaluation(
            device_str=args.device,
            binder_seq=args.binder_seq,
            target_seq=args.target_seq,
            label=args.label,
            design_entry_id=args.design_entry_id,
            entry_ids=entry_ids,
            n_repeats=args.n_repeats,
            N_cycle=args.n_cycles,
        )

    elif args.eval_mode == "all":
        logger.info("=" * 70)
        logger.info("模式: all（zeroth + full 依次运行）")
        logger.info("=" * 70)
        run_zeroth_evaluation(args.device, entry_ids)
        logger.info("")
        run_full_ode_evaluation(
            device_str=args.device,
            design_entry_id=args.design_entry_id,
            entry_ids=entry_ids,
            n_repeats=args.n_repeats,
            N_cycle=args.n_cycles,
        )

    logger.info(f"\n日志文件: {log_filepath}")


if __name__ == "__main__":
    main()
