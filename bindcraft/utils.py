"""
BindCraft v2 - 基础工具函数

提供通用的辅助函数，供其他模块使用
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Union


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


def _build_features_from_sequences(
    seq_input: TargetSeqInput,
    device: torch.device,
    name: str,
) -> Tuple[Dict, int, int]:
    """
    基于一条或多条蛋白链序列构建 Protenix 输入特征。
    """
    from protenix.data.inference.json_to_feature import SampleDictToFeatures
    from protenix.data.inference.infer_dataloader import InferenceTemplateFeaturizer
    from protenix.data.utils import data_type_transform, make_dummy_feature

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

    template_features = InferenceTemplateFeaturizer.make_template_feature(
        bioassembly=sample_dict["sequences"],
        atom_array=atom_array,
        use_template=False,
        online_template_featurizer=None,
    )

    dummy_feats = ["msa"]
    if len(template_features) == 0:
        dummy_feats.append("template")
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
) -> Tuple[Dict, int, int]:
    """
    构建 Binder-Target 复合物的输入特征，支持单链或多链 target。
    
    Args:
        binder_seq: Binder 单字母序列
        target_seq: Target 单字母序列，或多条 target 链组成的列表/逗号分隔字符串
        device: 计算设备
        
    Returns:
        input_feature_dict: 输入特征字典
        N_token: Token 总数
        N_atom: 原子总数
    """
    target_seqs = _normalize_chain_sequences(target_seq)
    seqs = [binder_seq] + target_seqs
    return _build_features_from_sequences(seqs, device, name="binder_target_complex")


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
