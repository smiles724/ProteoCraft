"""
BindCraft v2 - 基础工具函数

提供通用的辅助函数，供其他模块使用
"""

import torch
import numpy as np
from typing import Dict, Tuple, List


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

def build_single_chain_features(seq: str, device: torch.device) -> Tuple[Dict, int, int]:
    """
    构建单链蛋白质的输入特征（用于 W_AAE 预计算和 s_target 提取）

    与 build_complex_features 的区别：只有一条链，无 context 污染。

    Args:
        seq: 单字母蛋白质序列
        device: 计算设备

    Returns:
        input_feature_dict: 输入特征字典
        N_token: Token 总数（= len(seq)）
        N_atom: 原子总数
    """
    from protenix.data.inference.json_to_feature import SampleDictToFeatures
    from protenix.data.inference.infer_dataloader import InferenceTemplateFeaturizer
    from protenix.data.utils import data_type_transform, make_dummy_feature

    sample_dict = {
        "sequences": [
            {"proteinChain": {"sequence": seq, "count": 1}},
        ],
        "name": "single_chain"
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


def build_complex_features(binder_seq: str, target_seq: str, device: torch.device) -> Tuple[Dict, int, int]:
    """
    构建 Binder-Target 复合物的输入特征
    
    Args:
        binder_seq: Binder 单字母序列
        target_seq: Target 单字母序列
        device: 计算设备
        
    Returns:
        input_feature_dict: 输入特征字典
        N_token: Token 总数
        N_atom: 原子总数
    """
    from protenix.data.inference.json_to_feature import SampleDictToFeatures
    from protenix.data.inference.infer_dataloader import InferenceTemplateFeaturizer
    from protenix.data.utils import data_type_transform, make_dummy_feature
    
    # 构建 sample_dict
    sample_dict = {
        "sequences": [
            {"proteinChain": {"sequence": binder_seq, "count": 1}},
            {"proteinChain": {"sequence": target_seq, "count": 1}},
        ],
        "name": "binder_target_complex"
    }
    
    # 转换为特征
    sample2feat = SampleDictToFeatures(sample_dict)
    features_dict, atom_array, token_array = sample2feat.get_feature_dict()
    features_dict["distogram_rep_atom_mask"] = torch.Tensor(
        atom_array.distogram_rep_atom_mask
    ).long()
    
    # 添加 template 特征（空）
    template_features = InferenceTemplateFeaturizer.make_template_feature(
        bioassembly=sample_dict["sequences"],
        atom_array=atom_array,
        use_template=False,
        online_template_featurizer=None,
    )
    
    # 添加 dummy MSA 和 template
    dummy_feats = ["msa"]
    if len(template_features) == 0:
        dummy_feats.append("template")
    features_dict = make_dummy_feature(features_dict=features_dict, dummy_feats=dummy_feats)
    
    # 类型转换
    feat = data_type_transform(feat_or_label_dict=features_dict)
    
    # 移动到设备
    input_feature_dict = {}
    for k, v in feat.items():
        if isinstance(v, torch.Tensor):
            input_feature_dict[k] = v.to(device)
        else:
            input_feature_dict[k] = v
    
    N_token = feat["token_index"].shape[0]
    N_atom = feat["atom_to_token_idx"].shape[0]
    
    return input_feature_dict, N_token, N_atom


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
    "i_con":   1.0,
    "iptm":    0.05,
    "rg":      0.3,
    "helix":  -0.3,
    "clash":   0.5,
    "i_clash": 0.5,
    "pde":     0.1,
    "i_pde":   0.05,
}

DEFAULT_CONTACT_PARAMS = {
    "intra_cutoff":  14.0,
    "inter_cutoff":  20.0,
    "intra_num":     2,
    "inter_num":     2,
    "seqsep":        9,
}

DEFAULT_CLASH_PARAMS = {
    "r_clash":     2.0,
    "bond_sep":    1,
}
