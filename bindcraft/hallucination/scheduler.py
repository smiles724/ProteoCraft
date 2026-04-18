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


class MacroBreathTracker:
    """
    宏观呼吸机制：当混合优化陷入局部最优时，主动跳出。

    策略：
      - 跟踪零阶 loss 的历史，当连续 P 步均未低于当前最佳时触发"呼吸"状态
      - 呼吸状态下，tau 回退至宏观拓扑阶段上限（约 0.9），同时临时降低 con/i_con 权重
        以放宽接触约束，允许骨架做大幅拓扑调整
      - 连续 R 步后恢复正常调度（或若 loss 已重新下降则提前退出）
      - 与模拟退火（Step 4 接受准则）形成双保险

    Args:
        patience:       触发呼吸的连续停滞步数（推荐 6~12）
        tau_recovery:   回退目标 tau 值（宏观拓扑上限附近）
        con_factor:     呼吸状态下 con/i_con 权重的缩减系数（乘以此系数）
        recovery_steps: 呼吸状态持续步数（达到后恢复正常调度）
    """

    def __init__(
        self,
        patience: int = 8,
        tau_recovery: float = 0.9,
        con_factor: float = 0.3,
        recovery_steps: int = 3,
    ):
        self.patience = patience
        self.tau_recovery = tau_recovery
        self.con_factor = con_factor
        self.recovery_steps = recovery_steps

        self._best_loss = float("inf")
        self._no_improve_steps = 0
        self._breathing_countdown = 0       # 剩余呼吸步数
        self._forced_phase_override = False  # 呼吸期间强制回到宏观阶段

    def update(self, zeroth_loss: float) -> None:
        """
        每轮零阶验证后调用，检测是否需要触发或退出呼吸状态。
        """
        if self._breathing_countdown > 0:
            # 正在呼吸：检查 loss 是否已重新开始下降
            if zeroth_loss < self._best_loss - 1e-4:
                logger.debug(
                    f"MacroBreath: loss improved during breathing "
                    f"({zeroth_loss:.4f} < {self._best_loss:.4f}), exiting early"
                )
                self._breathing_countdown = 0
            else:
                self._breathing_countdown -= 1
                if self._breathing_countdown == 0:
                    logger.info(
                        f"MacroBreath: recovery completed. "
                        f"Best loss during cycle: {self._best_loss:.4f}"
                    )
            # 无论如何都更新 best（下降就更新）
            if zeroth_loss < self._best_loss:
                self._best_loss = zeroth_loss
            return

        # 非呼吸状态：检测停滞
        if zeroth_loss < self._best_loss - 1e-4:
            self._best_loss = zeroth_loss
            self._no_improve_steps = 0
        else:
            self._no_improve_steps += 1
            if self._no_improve_steps >= self.patience:
                logger.warning(
                    f"MacroBreath: stagnation detected "
                    f"(no improvement for {self._no_improve_steps} steps, "
                    f"best={self._best_loss:.4f}, current={zeroth_loss:.4f}), "
                    f"triggering macro-breathing (tau -> {self.tau_recovery}, "
                    f"con/i_con -> x{self.con_factor}, {self.recovery_steps} steps)"
                )
                self._breathing_countdown = self.recovery_steps
                self._no_improve_steps = 0

    @property
    def is_breathing(self) -> bool:
        """当前是否处于呼吸回退状态。"""
        return self._breathing_countdown > 0

    def get_breathing_tau(self) -> float:
        """呼吸状态下返回强制 tau 值（无论当前步数如何）。"""
        return self.tau_recovery

    def apply_breathing_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        呼吸状态下返回临时调整后的 loss 权重（降低 con/i_con）。

        Returns:
            新的权重字典（与输入独立，不修改原字典）
        """
        adjusted = dict(weights)
        for key in ("con", "i_con"):
            if key in adjusted:
                adjusted[key] = weights[key] * self.con_factor
        return adjusted


# ============================================================
# TauScheduler（Coarse-to-Fine 时间步调度）
# ============================================================

class TauScheduler:
    """
    Coarse-to-Fine 时间步调度器。

    三阶段调度（对应算法指导2.md 第七章第2节）：
      阶段 1 (0%-30%):   tau in [16.0, 8.0]  -- 宏观拓扑探索
      阶段 2 (30%-70%):  tau in [8.0, 2.0]   -- 界面雏形塑造
      阶段 3 (70%-100%): tau in [2.0, 0.1]   -- 原子级细化

    tau 在每个阶段内线性衰减（从 high 到 low），可叠加随机抖动。

    ─────────────────────────────────────────────────────────────────────────────
    tau 的 EDM 映射（sigma_data = 16.0）
    ─────────────────────────────────────────────────────────────────────────────

    Protenix 的扩散模块采用 EDM 框架，传入的 tau 即为 sigma。给定 sigma_data=16.0：

        s_ratio = sigma / sigma_data = tau / 16.0
        c_skip  = 1 / (1 + s_ratio²) = sigma_data² / (sigma_data² + sigma²)
        c_out   = sigma / sqrt(1 + s_ratio²)

    还原公式：x_denoised = c_skip * x_noisy + c_out * r_update

    tau 与 c_skip 的对应关系：

        tau=16.0  →  c_skip=0.50   →  网络预测占一半，原始坐标占一半
        tau=8.0   →  c_skip=0.80   →  网络预测主导
        tau=4.0   →  c_skip=0.94   →  适度调整
        tau=2.0   →  c_skip=0.98   →  微调
        tau=0.1   →  c_skip=0.9996  →  几乎跟随原始坐标
    """

    PHASES = [
        {"start": 0.0,  "end": 0.3,  "tau_high": 16.0,  "tau_low": 8.0},
        {"start": 0.3,  "end": 0.7,  "tau_high": 8.0,   "tau_low": 2.0},
        {"start": 0.7,  "end": 1.0,  "tau_high": 2.0,   "tau_low": 0.1},
    ]

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

    def get_dynamic_weights(
        self,
        step: int,
        total_steps: int,
        base_weights: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        """
        返回当前步使用的 loss 权重。

        当前仍不启用分阶段动态调权，因此返回 base_weights；若未提供则退回 DEFAULT_LOSS_WEIGHTS。
        这样可保留调用方运行时注入的额外权重（如 hotspot）。
        """
        weights = dict(DEFAULT_LOSS_WEIGHTS)
        if base_weights is not None:
            weights.update(base_weights)
        return weights

    def get_phase_name(self, step: int, total_steps: int) -> str:
        names = ["macro_topology", "interface_shaping", "atomic_refinement"]
        if total_steps <= 1:
            return names[1]
        progress = step / (total_steps - 1)
        return names[self._get_phase(progress)]
