#!/usr/bin/env python3
"""
BindCraft v2 - 入口脚本

使用方法（纯一阶模式）:
    python scripts/run_hallucination.py \
        --target GIVEQCCTSICSLYQLENYCN \
        --binder_length 40 \
        --n_steps 200 \
        --learning_rate 0.05 \
        --tau_noise 4.0

使用方法（混合 1st+0th 模式）:
    python scripts/run_hallucination.py \
        --target GIVEQCCTSICSLYQLENYCN \
        --binder_length 40 \
        --mode hybrid \
        --n_steps 100 \
        --learning_rate 0.05 \
        --tau_noise 4.0 \
        --k_top 10 \
        --b_candidates 8 \
        --t_sample 1.0 \
        --max_mutations_per_seq 3 \
        --use_annealing true \
        --t_init 1.0 \
        --t_min 0.01 \
        --use_tau_scheduling true \
        --tau_jitter true \
        --lambda_dist 2.0 \
        --lambda_clash 1.0 \
        --lambda_iptm -1.0 \
        --lambda_plddt -0.5
"""

import os
import sys
import argparse
import logging
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from bindcraft.hallucination.pipeline import run_hallucination, run_hybrid_hallucination


def setup_logging(log_file: str = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)


def load_protenix_model(model_name: str = "protenix_base_default_v1.0.0", device: str = "cuda"):
    from collections.abc import Mapping
    from protenix.config.config import parse_configs
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.model.protenix import Protenix

    logging.info(f"加载模型: {model_name}")

    checkpoint_path = f"/mnt/shared-storage-gpfs2/liyanchao2/model/checkpoint/{model_name}.pt"

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    base_configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    model_specific = model_configs[model_name]
    deep_update(base_configs, model_specific)

    arg_str = (
        f"--model_name {model_name} "
        f"--use_msa false "
        f"--dtype fp32 "
        f"--load_checkpoint_dir /mnt/shared-storage-gpfs2/liyanchao2/model/checkpoint "
        f"--dump_dir /tmp/bindcraft_hallucination "
    )
    configs = parse_configs(
        configs=base_configs, arg_str=arg_str, fill_required_with_null=True,
    )

    device_obj = torch.device(device)
    model = Protenix(configs).to(device_obj)

    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    state_dict = checkpoint["model"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    logging.info(f"模型加载完成")
    return model, configs


def main():
    parser = argparse.ArgumentParser(description="BindCraft v2 — Hallucination Pipeline")

    # === 基础参数 ===
    parser.add_argument("--target", type=str, required=True,
                       help="Target 序列（单字母氨基酸序列）")
    parser.add_argument("--mode", type=str, default="first_order",
                       choices=["first_order", "hybrid"],
                       help="优化模式: first_order (纯一阶) 或 hybrid (1st+0th 混合)")

    # === Binder 与优化参数 ===
    parser.add_argument("--binder_length", type=int, default=40,
                       help="Binder 长度（默认：40）")
    parser.add_argument("--n_steps", type=int, default=200,
                       help="优化步数（默认：200）")
    parser.add_argument("--learning_rate", type=float, default=0.05,
                       help="Adam 学习率（默认：0.05）")
    parser.add_argument("--tau_noise", type=float, default=4.0,
                       help="扩散噪声时间步（默认：4.0）")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备（默认：cuda）")
    parser.add_argument("--model_name", type=str, default="protenix_base_default_v1.0.0",
                       help="模型名称")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="输出目录（默认：./outputs）")
    parser.add_argument("--log_file", type=str, default=None,
                       help="日志文件路径")

    # === 混合模式专用参数 ===
    parser.add_argument("--k_top", type=int, default=10,
                       help="Top-K 突变候选数（默认：10）")
    parser.add_argument("--b_candidates", type=int, default=8,
                       help="零阶验证候选序列数（默认：8）")
    parser.add_argument("--t_sample", type=float, default=1.0,
                       help="突变采样温度（默认：1.0）")
    parser.add_argument("--max_mutations_per_seq", type=int, default=3,
                       help="每条序列的最大突变数（默认：3）")
    parser.add_argument("--use_annealing", type=lambda x: x.lower() == "true", default=True,
                       help="是否启用模拟退火（默认：true）")
    parser.add_argument("--t_init", type=float, default=1.0,
                       help="退火初始温度（默认：1.0）")
    parser.add_argument("--t_min", type=float, default=0.01,
                       help="退火最低温度（默认：0.01）")
    parser.add_argument("--use_tau_scheduling", type=lambda x: x.lower() == "true", default=True,
                       help="是否启用时间步退火调度（默认：true）")
    parser.add_argument("--tau_jitter", type=lambda x: x.lower() == "true", default=True,
                       help="是否启用时间步抖动（默认：true）")

    # === 损失函数权重（key 必须与 DEFAULT_LOSS_WEIGHTS 一致） ===
    parser.add_argument("--w_plddt", type=float, default=None, help="pLDDT 权重")
    parser.add_argument("--w_pae", type=float, default=None, help="pAE (intra) 权重")
    parser.add_argument("--w_i_pae", type=float, default=None, help="i_pAE (inter) 权重")
    parser.add_argument("--w_con", type=float, default=None, help="con (intra contact) 权重")
    parser.add_argument("--w_i_con", type=float, default=None, help="i_con (inter contact) 权重")
    parser.add_argument("--w_iptm", type=float, default=None, help="iPTM 权重")
    parser.add_argument("--w_rg", type=float, default=None, help="Rg 权重")
    parser.add_argument("--w_helix", type=float, default=None, help="Helix 权重")
    parser.add_argument("--w_clash", type=float, default=None, help="clash (intra) 权重")
    parser.add_argument("--w_i_clash", type=float, default=None, help="i_clash (inter) 权重")
    parser.add_argument("--w_pde", type=float, default=None, help="PDE (intra) 权重")
    parser.add_argument("--w_i_pde", type=float, default=None, help="i_PDE (inter) 权重")

    args = parser.parse_args()

    setup_logging(args.log_file)
    os.makedirs(args.output_dir, exist_ok=True)

    model, configs = load_protenix_model(args.model_name, args.device)

    # 组装 loss_weights: 仅覆盖 CLI 显式指定的项，其余保持 DEFAULT_LOSS_WEIGHTS
    loss_weights = {}
    weight_arg_map = {
        "plddt": args.w_plddt, "pae": args.w_pae, "i_pae": args.w_i_pae,
        "con": args.w_con, "i_con": args.w_i_con, "iptm": args.w_iptm,
        "rg": args.w_rg, "helix": args.w_helix,
        "clash": args.w_clash, "i_clash": args.w_i_clash,
        "pde": args.w_pde, "i_pde": args.w_i_pde,
    }
    for key, val in weight_arg_map.items():
        if val is not None:
            loss_weights[key] = val

    if args.mode == "first_order":
        best_seq, final_seq, losses_history = run_hallucination(
            model=model,
            target_seq=args.target,
            L_binder=args.binder_length,
            N_steps=args.n_steps,
            learning_rate=args.learning_rate,
            tau_noise=args.tau_noise,
            device=torch.device(args.device),
            loss_weights=loss_weights,
        )
    else:
        best_seq, final_seq, losses_history = run_hybrid_hallucination(
            model=model,
            target_seq=args.target,
            L_binder=args.binder_length,
            N_steps=args.n_steps,
            learning_rate=args.learning_rate,
            tau_noise=args.tau_noise,
            device=torch.device(args.device),
            loss_weights=loss_weights,
            K_top=args.k_top,
            B_candidates=args.b_candidates,
            T_sample=args.t_sample,
            max_mutations_per_seq=args.max_mutations_per_seq,
            use_annealing=args.use_annealing,
            T_init=args.t_init,
            T_min=args.t_min,
            use_tau_scheduling=args.use_tau_scheduling,
            tau_jitter=args.tau_jitter,
        )

    output_file = os.path.join(args.output_dir, "hallucination_result.txt")
    with open(output_file, "w") as f:
        f.write(f"Target: {args.target}\n")
        f.write(f"Binder Length: {args.binder_length}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Optimization Steps: {args.n_steps}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Tau Noise: {args.tau_noise}\n")
        f.write(f"\n")
        f.write(f"Best Sequence: {best_seq}\n")
        f.write(f"Final Sequence: {final_seq}\n")
        f.write(f"\n")
        f.write(f"Loss History:\n")
        for i, loss in enumerate(losses_history):
            f.write(f"Step {i}: {loss:.6f}\n")

    logging.info(f"结果已保存到: {output_file}")

    print("\n" + "=" * 70)
    print(" Hallucination 完成！")
    print(f" 最佳序列: {best_seq}")
    print(f" 结果文件: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
