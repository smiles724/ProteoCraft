#!/usr/bin/env python3
"""
BindCraft v2 - 入口脚本

运行 Hallucination 优化，设计 Binder 序列

使用方法：
    python scripts/run_hallucination.py --target GIVEQCCTSICSLYQLENYCN --binder_length 40
"""

import os
import sys
import argparse
import logging
import torch

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
PROTENIX_ROOT = PROJECT_ROOT  # Protenix_BindCraft 本身就是 Protenix 的副本
sys.path.insert(0, PROTENIX_ROOT)

from bindcraft.hallucination.pipeline import run_hallucination


# ============================================================
# 日志配置
# ============================================================

def setup_logging(log_file: str = None):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)


# ============================================================
# 加载 Protenix 模型
# ============================================================

def load_protenix_model(model_name: str = "protenix_base_default_v1.0.0", device: str = "cuda"):
    """
    加载 Protenix 模型
    
    Args:
        model_name: 模型名称
        device: 计算设备
        
    Returns:
        model: 加载的模型
        configs: 配置
    """
    from collections.abc import Mapping
    from protenix.config.config import parse_configs
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.model.protenix import Protenix
    
    logging.info(f"加载模型: {model_name}")
    
    checkpoint_path = f"/hpc2hdd/home/bduan436/checkpoint/{model_name}.pt"
    
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    # 合并配置
    base_configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    model_specific = model_configs[model_name]
    deep_update(base_configs, model_specific)
    
    # 解析配置
    arg_str = (
        f"--model_name {model_name} "
        f"--use_msa false "
        f"--dtype fp32 "
        f"--load_checkpoint_dir /hpc2hdd/home/bduan436/checkpoint "
        f"--dump_dir /tmp/bindcraft_hallucination "
    )
    configs = parse_configs(
        configs=base_configs, arg_str=arg_str, fill_required_with_null=True,
    )
    
    # 创建模型
    device_obj = torch.device(device)
    model = Protenix(configs).to(device_obj)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    state_dict = checkpoint["model"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    logging.info(f"模型加载完成")
    
    return model, configs


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="BindCraft v2 - Hallucination Pipeline")
    
    # 必需参数
    parser.add_argument("--target", type=str, required=True,
                       help="Target 序列（单字母）")
    
    # 可选参数
    parser.add_argument("--binder_length", type=int, default=40,
                       help="Binder 长度（默认：40）")
    parser.add_argument("--n_steps", type=int, default=200,
                       help="优化步数（默认：200）")
    parser.add_argument("--learning_rate", type=float, default=0.05,
                       help="学习率（默认：0.05）")
    parser.add_argument("--tau_noise", type=float, default=4.0,
                       help="噪声时间步（默认：4.0）")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备（默认：cuda）")
    parser.add_argument("--model_name", type=str, default="protenix_base_default_v1.0.0",
                       help="模型名称")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="输出目录")
    parser.add_argument("--log_file", type=str, default=None,
                       help="日志文件路径")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_file)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model, configs = load_protenix_model(args.model_name, args.device)
    
    # 运行 Hallucination
    best_seq, final_seq, losses_history = run_hallucination(
        model=model,
        target_seq=args.target,
        L_binder=args.binder_length,
        N_steps=args.n_steps,
        learning_rate=args.learning_rate,
        tau_noise=args.tau_noise,
        device=torch.device(args.device),
    )
    
    # 保存结果
    output_file = os.path.join(args.output_dir, "hallucination_result.txt")
    with open(output_file, "w") as f:
        f.write(f"Target: {args.target}\n")
        f.write(f"Binder Length: {args.binder_length}\n")
        f.write(f"Optimization Steps: {args.n_steps}\n")
        f.write(f"\n")
        f.write(f"Best Sequence: {best_seq}\n")
        f.write(f"Final Sequence: {final_seq}\n")
        f.write(f"\n")
        f.write(f"Loss History:\n")
        for i, loss in enumerate(losses_history):
            f.write(f"Step {i}: {loss:.6f}\n")
    
    logging.info(f"结果已保存到: {output_file}")
    
    print("\n" + "=" * 70)
    print("✅ Hallucination 完成！")
    print(f"最佳序列: {best_seq}")
    print(f"结果文件: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
