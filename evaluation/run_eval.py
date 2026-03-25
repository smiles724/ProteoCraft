#!/usr/bin/env python3
"""
Retrodiction 评估 Pipeline - Step 3: 零阶全链路评估

四组序列（designed / random / real_vh / real_full）全部通过 zeroth_order_eval() AF3 推断，
用同一把尺子（iPTM / pLDDT / total_loss）打分。

  - designed / random: BindCraft v2 设计的 40aa binder 及等长随机对照
  - real_vh: 仅重链（~115aa），与 designed/random 长度最接近，可做长度接近的公平对比
  - real_full: 完整 H+L 链（~430aa），与 ab-raw.csv DockQ 参考条件一致

DockQ 参考列仅附加到 real_full 组（因为 ab-raw.csv 中的 DockQ 是在完整 H+L+Antigen 复合物上计算的）。

输出: evaluation/outputs/eval_results.csv

用法:
    CUDA_VISIBLE_DEVICES=3 python evaluation/run_eval.py [--device cuda:0]
"""

import os
import sys
import csv
import json
import glob
import time
import logging
from datetime import datetime

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.configs import (
    OUTPUT_DIR, DESIGNS_DIR, LOG_DIR, MODEL_NAME,
    AB_RAW_CSV, AB_RAW_MODEL_PREFIX, EVAL_MAX_TOTAL_SEQ_LEN,
)

os.makedirs(LOG_DIR, exist_ok=True)
log_filepath = os.path.join(
    LOG_DIR, f"retrodiction_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

RESULT_COLUMNS = [
    "entry_id", "seq_type", "binder_length", "target_length",
    "total_loss", "plddt", "pae", "i_pae", "con", "i_con", "iptm", "rg", "helix",
    "mean_plddt_conf", "iptm_conf", "ptm_conf",
    "dockq_ref", "lddt_ref",
    "eval_time_sec",
]


def load_dockq_reference(ab_raw_path: str, model_prefix: str) -> dict:
    """
    从 ab-raw.csv 读取每个 entry_id 在指定模型下的 best-DockQ，
    仅作为参考列附在 real 组结果中。

    返回 {entry_id: {dockq, lddt}}
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
                best[eid] = {
                    "dockq": dq,
                    "lddt": float(row.get("lddt", 0.0)),
                }
    return best


def eval_one_sequence(
    binder_seq: str,
    target_seq: str,
    model,
    device: torch.device,
):
    """调用 zeroth_order_eval 并返回扁平化的指标字典。"""
    from bindcraft.hallucination.zeroth_order import zeroth_order_eval

    t0 = time.time()
    result = zeroth_order_eval(
        binder_seq=binder_seq,
        target_seq=target_seq,
        model=model,
        device=device,
    )
    elapsed = time.time() - t0

    row = {
        "total_loss": result.total_loss,
        "eval_time_sec": round(elapsed, 1),
    }
    for key in ["plddt", "pae", "i_pae", "con", "i_con", "iptm", "rg", "helix"]:
        row[key] = result.loss_dict.get(key, 0.0)

    sc = result.summary_confidence
    row["mean_plddt_conf"] = sc.get("plddt", sc.get("mean_plddt", 0.0))
    row["iptm_conf"] = sc.get("iptm", 0.0)
    row["ptm_conf"] = sc.get("ptm", 0.0)

    return row


def get_real_binder_for_eval(
    data: dict, target_len: int, binder_key: str
) -> str | None:
    """
    从 design JSON 中提取指定键的 real binder 序列。
    若 target + binder 总长超过 EVAL_MAX_TOTAL_SEQ_LEN 则返回 None（跳过推断）。

    Args:
        data: design JSON 内容
        target_len: target 序列长度
        binder_key: JSON 中的字段名，如 "real_binder_heavy" 或 "real_binder_seq"
    """
    binder = data.get(binder_key, "")
    if not binder:
        return None
    if target_len + len(binder) > EVAL_MAX_TOTAL_SEQ_LEN:
        return None
    return binder


def run_evaluation(device_str: str, entry_ids: list = None):
    from scripts.run_hallucination import load_protenix_model

    design_files = sorted(glob.glob(os.path.join(DESIGNS_DIR, "*.json")))
    if not design_files:
        logger.error(f"未找到设计结果: {DESIGNS_DIR}/*.json")
        return

    if entry_ids:
        design_files = [f for f in design_files
                        if any(eid in f for eid in entry_ids)]
        logger.info(f"按 --entry_ids 过滤后: {len(design_files)} 个设计结果待评估")

    logger.info(f"共 {len(design_files)} 个设计结果待评估")

    logger.info(f"加载 ab-raw.csv DockQ 参考 ({AB_RAW_MODEL_PREFIX}) ...")
    dockq_ref = load_dockq_reference(AB_RAW_CSV, AB_RAW_MODEL_PREFIX)
    logger.info(f"  共 {len(dockq_ref)} 个 entry 有 DockQ 参考")

    device = torch.device(device_str)
    logger.info(f"加载模型 {MODEL_NAME} -> {device}")
    model, configs = load_protenix_model(MODEL_NAME, str(device))

    out_path = os.path.join(OUTPUT_DIR, "eval_results.csv")
    file_exists = os.path.exists(out_path)
    mode = "a" if file_exists else "w"
    with open(out_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
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
            "random": data["random_seq"],
        }

        # real_vh: 仅重链（~115aa），与 designed/random 长度最接近
        real_vh_binder = get_real_binder_for_eval(data, target_len, "real_binder_heavy")
        if real_vh_binder:
            sequences["real_vh"] = real_vh_binder
        else:
            heavy_raw = data.get("real_binder_heavy", "")
            if heavy_raw:
                logger.warning(
                    f"  {entry_id}: target({target_len}) + VH({len(heavy_raw)}) = "
                    f"{target_len + len(heavy_raw)}aa > {EVAL_MAX_TOTAL_SEQ_LEN}，跳过 real_vh 组"
                )
            else:
                logger.warning(f"  {entry_id}: 无 real_binder_heavy，跳过 real_vh 组")

        # real_full: 完整 H+L 链（~430aa），与 ab-raw.csv DockQ 参考条件一致
        real_full_binder = get_real_binder_for_eval(data, target_len, "real_binder_seq")
        if real_full_binder:
            sequences["real_full"] = real_full_binder
        else:
            full_raw = data.get("real_binder_seq", "")
            if full_raw:
                logger.warning(
                    f"  {entry_id}: target({target_len}) + H+L({len(full_raw)}) = "
                    f"{target_len + len(full_raw)}aa > {EVAL_MAX_TOTAL_SEQ_LEN}，跳过 real_full 组"
                )
            else:
                logger.warning(f"  {entry_id}: 无 real_binder_seq，跳过 real_full 组")

        logger.info("=" * 60)
        logger.info(
            f"[{idx+1}/{len(design_files)}] 评估 {entry_id} | "
            f"target={target_len}aa | {len(sequences)} 组序列"
        )

        gt = dockq_ref.get(entry_id)

        for seq_type, binder_seq in sequences.items():
            logger.info(
                f"  {seq_type}: {binder_seq[:20]}... ({len(binder_seq)}aa) [AF3 推断]"
            )
            try:
                metrics = eval_one_sequence(binder_seq, target_seq, model, device)
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
                writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
                writer.writerow(row)

            total_evals += 1

            log_line = (
                f"    total_loss={metrics['total_loss']:.4f} | "
                f"plddt_conf={metrics['mean_plddt_conf']:.1f} | "
                f"iptm_conf={metrics['iptm_conf']:.3f} | "
                f"ptm_conf={metrics['ptm_conf']:.3f} | "
                f"{metrics['eval_time_sec']}s"
            )
            if seq_type == "real_full" and gt:
                log_line += f" | DockQ_ref={gt['dockq']:.4f}"
            logger.info(log_line)

    logger.info("=" * 60)
    logger.info(f"评估完成 | 共 {total_evals} 次推断 | 结果: {out_path}")
    logger.info(f"日志: {log_filepath}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="零阶全链路评估")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--entry_ids", type=str, default=None,
        help="逗号分隔的 entry_id 列表，如 '8trt,9ho4'。指定时只评估这些 case。"
    )
    args = parser.parse_args()
    entry_ids = None
    if args.entry_ids:
        entry_ids = [e.strip() for e in args.entry_ids.split(",") if e.strip()]
    run_evaluation(args.device, entry_ids)


if __name__ == "__main__":
    main()
