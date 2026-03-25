"""
BindCraft v2 - 基于 AlphaFold 3 的蛋白质序列逆向设计

核心模块：
- hallucination: STE 桥接、单步外推、主循环
- losses: 几何 Loss 函数
- utils: 通用工具函数
"""

__version__ = "2.0.0"
__author__ = "BindCraft Team"

from . import hallucination
from . import losses
from . import utils

__all__ = ["hallucination", "losses", "utils"]
