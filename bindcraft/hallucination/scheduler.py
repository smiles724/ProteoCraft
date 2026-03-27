"""
BindCraft v2 - 调度器模块

实现两个核心调度机制：
1. AnnealingScheduler：模拟退火接受准则，以 Metropolis-Hastings 概率接受略差候选
2. TauScheduler：Coarse-to-Fine 时间步调度 + 动态 loss 权重 + 时间步抖动

对应指导文档：
  - 算法指导2.md 第六章 Step 4（退火接受）
  - 算法指导2.md 第七章（tau 调度 + jittering）
  - Bindcraft v2零阶更新.md 3.1 节（微观退火 + 宏观呼吸）
  - Bindcraft v2零阶更新.md 5.2 节（动态 loss 权重随 tau 调整）
"""

import math
import random
import logging
from typing import Dict

from bindcraft.utils import DEFAULT_LOSS_WEIGHTS

logger = logging.getLogger(__name__)


class AnnealingScheduler:
    """
    模拟退火温度调度器。

    P_accept = min(1, exp((L_current - L_candidate) / T))

    当 candidate_loss < current_loss 时必定接受；
    当 candidate_loss >= current_loss 时以概率 P_accept 接受，
    允许系统跳出局部最优。

    温度 T 从 T_init 指数衰减到 T_min。
    """

    def __init__(
        self,
        T_init: float = 1.0,
        T_min: float = 0.01,
        decay: str = "exponential",
    ):
        self.T_init = T_init
        self.T_min = T_min
        self.decay = decay

    def get_temperature(self, step: int, total_steps: int) -> float:
        if total_steps <= 1:
            return self.T_init
        progress = step / (total_steps - 1)

        if self.decay == "exponential":
            log_ratio = math.log(max(self.T_min / self.T_init, 1e-10))
            T = self.T_init * math.exp(log_ratio * progress)
        elif self.decay == "linear":
            T = self.T_init + (self.T_min - self.T_init) * progress
        elif self.decay == "cosine":
            T = self.T_min + 0.5 * (self.T_init - self.T_min) * (1 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown decay type: {self.decay}")

        return max(T, self.T_min)

    def accept(
        self,
        current_loss: float,
        candidate_loss: float,
        step: int,
        total_steps: int,
    ) -> bool:
        """
        Metropolis-Hastings 接受准则。

        Returns:
            (accepted, was_probabilistic) -- 是否接受，以及是否是概率接受（而非严格改进）
        """
        if candidate_loss < current_loss:
            return True

        T = self.get_temperature(step, total_steps)
        delta = current_loss - candidate_loss  # negative when candidate is worse
        if T < 1e-12:
            return False

        p_accept = math.exp(delta / T)
        accepted = random.random() < p_accept

        if accepted:
            logger.info(
                f"  Annealing: probabilistic accept | "
                f"delta={delta:.4f} T={T:.4f} P={p_accept:.4f}"
            )
        return accepted


class TauScheduler:
    """
    Coarse-to-Fine 时间步调度器。

    三阶段调度（对应算法指导2.md 第七章第2节）：
      阶段 1 (0%-30%):   tau in [0.6, 0.9]  -- 宏观拓扑探索
      阶段 2 (30%-70%):  tau in [0.2, 0.6]  -- 界面雏形塑造
      阶段 3 (70%-100%): tau in [0.01, 0.2] -- 原子级细化

    tau 在每个阶段内线性衰减（从 high 到 low），可叠加随机抖动。
    同时根据当前 tau 阶段动态调整 loss 权重。
    """

    PHASES = [
        {"start": 0.0,  "end": 0.3,  "tau_high": 0.9,  "tau_low": 0.6},
        {"start": 0.3,  "end": 0.7,  "tau_high": 0.6,  "tau_low": 0.2},
        {"start": 0.7,  "end": 1.0,  "tau_high": 0.2,  "tau_low": 0.01},
    ]

    WEIGHT_PROFILES = {
        0: {
            "plddt": 0.02, "pae": 0.1, "i_pae": 0.02,
            "con": 1.5, "i_con": 1.5, "iptm": 0.01,
            "rg": 0.5, "helix": -0.3,
            "clash": 0.2, "i_clash": 0.2,
            "pde": 0.05, "i_pde": 0.02,
        },
        1: {
            "plddt": 0.1, "pae": 0.4, "i_pae": 0.1,
            "con": 1.0, "i_con": 1.0, "iptm": 0.05,
            "rg": 0.3, "helix": -0.3,
            "clash": 0.5, "i_clash": 0.5,
            "pde": 0.1, "i_pde": 0.05,
        },
        2: {
            "plddt": 0.2, "pae": 0.5, "i_pae": 0.2,
            "con": 0.8, "i_con": 0.8, "iptm": 0.1,
            "rg": 0.2, "helix": -0.3,
            "clash": 1.0, "i_clash": 1.0,
            "pde": 0.15, "i_pde": 0.1,
        },
    }

    def __init__(self, jitter: bool = True, jitter_scale: float = 0.1):
        self.jitter = jitter
        self.jitter_scale = jitter_scale

    def _get_phase(self, progress: float) -> int:
        for i, phase in enumerate(self.PHASES):
            if progress < phase["end"] or i == len(self.PHASES) - 1:
                return i
        return len(self.PHASES) - 1

    def get_tau(self, step: int, total_steps: int) -> float:
        if total_steps <= 1:
            return self.PHASES[1]["tau_high"]
        progress = step / (total_steps - 1)
        phase_idx = self._get_phase(progress)
        phase = self.PHASES[phase_idx]

        local_progress = (progress - phase["start"]) / max(phase["end"] - phase["start"], 1e-8)
        local_progress = max(0.0, min(1.0, local_progress))

        tau = phase["tau_high"] + (phase["tau_low"] - phase["tau_high"]) * local_progress

        if self.jitter:
            span = phase["tau_high"] - phase["tau_low"]
            noise = (random.random() - 0.5) * 2.0 * self.jitter_scale * span
            tau = max(phase["tau_low"], min(phase["tau_high"], tau + noise))

        return tau

    def get_dynamic_weights(self, step: int, total_steps: int) -> Dict[str, float]:
        """
        根据当前 tau 阶段返回动态 loss 权重。

        阶段 1（大 tau）：强调 con/i_con/rg，降低 plddt/iptm
        阶段 2（中 tau）：恢复默认权重
        阶段 3（小 tau）：强调 plddt/iptm/pae，略降 con
        """
        if total_steps <= 1:
            return dict(DEFAULT_LOSS_WEIGHTS)
        progress = step / (total_steps - 1)
        phase_idx = self._get_phase(progress)
        return dict(self.WEIGHT_PROFILES[phase_idx])

    def get_phase_name(self, step: int, total_steps: int) -> str:
        names = ["macro_topology", "interface_shaping", "atomic_refinement"]
        if total_steps <= 1:
            return names[1]
        progress = step / (total_steps - 1)
        return names[self._get_phase(progress)]
