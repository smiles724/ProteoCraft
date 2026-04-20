"""
BindCraft v2 - MCTS 控制器

MCTS（蒙特卡洛树搜索）作为三步曲引擎的宏观调度层，不改变三步曲内部任何代码，
只负责决定每轮迭代把 Step1+Step2+Step3 的计算资源分配给树上哪个历史序列节点。

对应文档：
    MCTS 融入 BindCraft v2 优化流程详解.md

核心概念：
- 树中所有节点都经过了 Step3 零阶验证
- UCB 从整棵历史树上选择节点，而非仅从当前代选择
- 被淘汰的序列存入树中，可被未来轮次重新选中
- UCB 公式自动实现 exploitation-exploration 平衡，隐式实现全局回退

UCB 公式：
    s* = argmax_{s_i in Tree} ( -W(s_i)/V(s_i) + C * sqrt(ln(N_total)/V(s_i)) )
    W(s_i) = 节点及子代在 Step3 中的累计真实 Loss 之和
    V(s_i) = 节点及子代被执行 Step3 的总次数
    N_total = 从 Phase 0 至今执行 Step3 的全局总次数
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """
    MCTS 树节点。

    每个节点对应一条经过 Step3 零阶验证的离散氨基酸序列。

    Attributes:
        node_id:        节点唯一标识符（全局自增）
        sequence:       离散氨基酸序列（单字母 str）
        logits:         连续 Logits [L_binder, 20]（用于重建 theta）
        coordinate:      Step3 产出的全原子结构 [1, N_atom, 3]（可为空，Phase0 根节点无）
        L_real:         该节点自身 Step3 零阶验证的真实 Loss
        W:              累计 Loss：自身及子代所有 Step3 验证的 L_real 之和（UCB 中的 W(s_i)）
        visit_count:    V(s_i)：自身及子代被执行 Step3 的总次数
        parent:         父节点引用（根节点为 None）
        children:       子节点列表
        depth:          节点在树中的深度（根节点深度 0）
    """
    node_id: int
    sequence: str
    logits: torch.Tensor                    # [L_binder, 20]
    coordinate: Optional[torch.Tensor]        # [1, N_atom, 3] 或 None
    L_real: float
    W: float = field(default=0.0)            # 初始化时 = L_real, V=1
    visit_count: int = field(default=0)     # 初始化时 = 1
    parent: Optional['MCTSNode'] = field(default=None, repr=False)
    children: List['MCTSNode'] = field(default_factory=list, repr=False)
    depth: int = field(default=0)

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_expanded(self) -> bool:
        """节点是否已有子节点（即是否被 expand 过）"""
        return len(self.children) > 0

    def get_path_to_root(self) -> List['MCTSNode']:
        """返回从该节点到根节点的路径（包含自身）"""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return path


class MCTSTree:
    """
    MCTS 树管理器。

    负责：
    1. UCB 选择：从整棵树上选择最优节点
    2. Expansion：将候选序列挂入树中
    3. Backpropagation：沿路径回传 Loss，更新 W 和 V
    4. 树剪枝：控制节点总数
    5. 全局状态追踪：N_total、根节点列表等
    """

    def __init__(
        self,
        C: float = 1.0,
        max_nodes: int = 5000,
        mcts_tau: float = 4.0,
    ):
        """
        Args:
            C:           探索常数（UCB 公式中的 C），控制 exploitation vs exploration 平衡
            max_nodes:   最大节点数，超出后剪掉 W/V 最差的叶子
            mcts_tau:    MCTS 模式下的固定 tau 值（不再采用退火调度）
        """
        self.C = C
        self.max_nodes = max_nodes
        self.mcts_tau = mcts_tau

        self._nodes: Dict[int, MCTSNode] = {}   # node_id -> MCTSNode
        self._next_id: int = 0                   # 全局自增 node_id
        self._roots: List[MCTSNode] = []         # 根节点列表
        self.N_total: int = 0                    # 全局 Step3 执行总次数

    def _new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def _register_node(self, node: MCTSNode) -> None:
        """将节点注册到树中"""
        self._nodes[node.node_id] = node

    # ── 根节点管理 ──────────────────────────────────────────────────────────

    def add_root_node(
        self,
        sequence: str,
        logits: torch.Tensor,
        L_real: float,
        coordinate: Optional[torch.Tensor] = None,
    ) -> MCTSNode:
        """
        添加一个根节点（Phase 0 冷启动时调用）。

        根节点初始化：W = L_real, V = 1
        N_total += 1
        """
        node = MCTSNode(
            node_id=self._new_id(),
            sequence=sequence,
            logits=logits.clone(),
            coordinate=coordinate.clone() if coordinate is not None else None,
            L_real=L_real,
            W=L_real,
            visit_count=1,
            parent=None,
            children=[],
            depth=0,
        )
        self._register_node(node)
        self._roots.append(node)
        self.N_total += 1
        logger.debug(
            f"[MCTS] Root node added: id={node.node_id} seq={sequence[:20]}... "
            f"L={L_real:.4f} N_total={self.N_total}"
        )
        return node

    @property
    def roots(self) -> List[MCTSNode]:
        return self._roots

    @property
    def total_nodes(self) -> int:
        return len(self._nodes)

    # ── UCB 核心 ────────────────────────────────────────────────────────────

    def ucb_score(self, node: MCTSNode) -> float:
        """
        计算节点的 UCB 分数。

        公式：
            score = -W(s_i)/V(s_i) + C * sqrt(ln(N_total)/V(s_i))

        - 利用项：-W/V，越低（取负后越大），越倾向继续深挖
        - 探索项：C * sqrt(ln(N_total)/V)，V 越小的节点探索奖励越大

        注意：ln(0) 在 N_total=0 时未定义，此时探索项设为极大值，
        确保每个节点在 Phase 0 后都有机会被探索。
        """
        if self.N_total == 0:
            # Phase 0 未开始时，探索项设为 inf
            exploration = float('inf')
        else:
            exploration = self.C * math.sqrt(math.log(self.N_total) / node.visit_count)

        exploitation = -node.W / node.visit_count
        return exploitation + exploration

    def select(self) -> MCTSNode:
        """
        UCB 选择：从整棵树上选择得分最高的节点。

        若选中已有子节点的内部节点，则以其序列为起点重新展开新分支。
        """
        if not self._nodes:
            raise RuntimeError("[MCTS] Tree is empty, cannot select")

        all_nodes = list(self._nodes.values())

        # 找 UCB 最高的节点
        best_node = max(all_nodes, key=lambda n: self.ucb_score(n))
        best_score = self.ucb_score(best_node)

        logger.debug(
            f"[MCTS] Selected: id={best_node.node_id} depth={best_node.depth} "
            f"seq={best_node.sequence[:20]}... "
            f"L_real={best_node.L_real:.4f} W={best_node.W:.4f} V={best_node.visit_count} "
            f"UCB={best_score:.4f} | tree_size={self.total_nodes} N_total={self.N_total}"
        )

        return best_node

    # ── Expansion ──────────────────────────────────────────────────────────

    def expand(
        self,
        parent: MCTSNode,
        candidates: List[Any],  # List[ZerothOrderResult]
    ) -> List[MCTSNode]:
        """
        将 K 个候选作为 parent 的子节点挂入树中。

        每个子节点初始化：
            W_k = L_real^{(k)}
            V_k = 1

        注意：这里只创建节点，不执行 backpropagation。
        Backpropagation 由 caller 在 expand 后统一调用。

        Args:
            parent:    父节点（由 select() 选出）
            candidates: K 个候选的零阶验证结果（ZerothOrderResult）

        Returns:
            新创建的子节点列表
        """
        if not candidates:
            return []

        new_nodes: List[MCTSNode] = []
        for cand in candidates:
            child = MCTSNode(
                node_id=self._new_id(),
                sequence=cand.binder_seq,
                logits=parent.logits.clone(),  # 从父节点继承，后续由 caller 同步
                coordinate=cand.coordinate[0:1].clone() if cand.coordinate is not None else None,
                L_real=cand.total_loss,
                W=cand.total_loss,     # 自身 Step3 验证的 Loss
                visit_count=1,         # 被 Step3 验证 1 次
                parent=parent,
                children=[],
                depth=parent.depth + 1,
            )
            self._register_node(child)
            parent.children.append(child)
            new_nodes.append(child)

            logger.debug(
                f"[MCTS] Expand: parent={parent.node_id} -> child={child.node_id} "
                f"seq={child.sequence[:20]}... L={child.L_real:.4f}"
            )

        return new_nodes

    # ── Backpropagation ────────────────────────────────────────────────────

    def backpropagate(
        self,
        node: MCTSNode,
        losses: List[float],
    ) -> None:
        """
        将 K 个子节点的 L_real 沿路径回传。

        对路径上每个祖先节点 s（从 node 到根）：
            W(s) <- W(s) + sum(L_real^{(k)})
            V(s) <- V(s) + K

        全局计数器同步更新：
            N_total <- N_total + K

        Args:
            node:   父节点（expand 的 parent，即 select 选出的节点）
            losses: K 个子节点的 L_real 列表
        """
        K = len(losses)
        sum_losses = sum(losses)

        # 从 node 到根，沿路径回传
        path = node.get_path_to_root()  # 包含 node 自身
        for ancestor in path:
            ancestor.W += sum_losses
            ancestor.visit_count += K

        # 全局计数器
        self.N_total += K

        logger.debug(
            f"[MCTS] Backprop: parent={node.node_id} K={K} sum_L={sum_losses:.4f} "
            f"N_total={self.N_total}"
        )

    # ── 树剪枝 ─────────────────────────────────────────────────────────────

    def prune(self) -> int:
        """
        树规模控制。

        当节点总数超过 max_nodes 时，删除 W/V 最差的叶子节点。
        保留所有根节点不被剪枝。

        Returns:
            被剪枝的节点数量
        """
        if self.total_nodes <= self.max_nodes:
            return 0

        n_prune = self.total_nodes - self.max_nodes

        # 收集所有叶子节点（且不是根节点）
        leaves = [
            n for n in self._nodes.values()
            if n.is_leaf() and not n.is_root()
        ]

        if not leaves:
            logger.warning(
                f"[MCTS] prune: no prunable leaves (only roots left, "
                f"size={self.total_nodes}/{self.max_nodes})"
            )
            return 0

        # 按 avg_loss = W/visit_count 从小到大排序（越低越"好"，越不应被剪）
        # 剪掉 W/V 最高的（即最差的）n_prune 个叶子
        leaves_sorted = sorted(leaves, key=lambda n: n.W / max(n.visit_count, 1), reverse=True)

        pruned = 0
        for leaf in leaves_sorted:
            if pruned >= n_prune:
                break
            self._remove_node(leaf)
            pruned += 1

        logger.info(
            f"[MCTS] Pruned {pruned} nodes (tree_size={self.total_nodes}/{self.max_nodes})"
        )
        return pruned

    def _remove_node(self, node: MCTSNode) -> None:
        """从树中删除一个节点（不检查是否为根）"""
        if node.parent is not None:
            if node in node.parent.children:
                node.parent.children.remove(node)
        else:
            # 是根节点
            if node in self._roots:
                self._roots.remove(node)
        del self._nodes[node.node_id]

    # ── 查询接口 ──────────────────────────────────────────────────────────

    def get_best_leaf(self) -> MCTSNode:
        """返回 W/V 最优的叶子节点（全局最优）"""
        leaves = [n for n in self._nodes.values() if n.is_leaf()]
        if not leaves:
            raise RuntimeError("[MCTS] No leaf nodes in tree")
        return min(leaves, key=lambda n: n.W / max(n.visit_count, 1))

    def get_best_root(self) -> MCTSNode:
        """返回 W/V 最优的根节点"""
        if not self._roots:
            raise RuntimeError("[MCTS] No root nodes")
        return min(self._roots, key=lambda n: n.L_real)

    def get_global_best(self) -> MCTSNode:
        """返回整棵树上 W/V 最优的节点"""
        all_nodes = list(self._nodes.values())
        return min(all_nodes, key=lambda n: n.W / max(n.visit_count, 1))

    def get_node_stats(self) -> Dict[str, Any]:
        """返回树的统计信息"""
        depths = [n.depth for n in self._nodes.values()]
        return {
            "total_nodes": self.total_nodes,
            "num_roots": len(self._roots),
            "num_leaves": sum(1 for n in self._nodes.values() if n.is_leaf()),
            "num_internal": sum(1 for n in self._nodes.values() if not n.is_leaf()),
            "N_total": self.N_total,
            "max_depth": max(depths) if depths else 0,
            "C": self.C,
            "max_nodes": self.max_nodes,
        }

    def log_tree_summary(self) -> None:
        """打印树的摘要信息"""
        stats = self.get_node_stats()
        best = self.get_global_best()
        logger.info(
            f"[MCTS] Tree summary: {stats['total_nodes']} nodes, "
            f"{stats['num_roots']} roots, {stats['num_leaves']} leaves, "
            f"max_depth={stats['max_depth']}, N_total={stats['N_total']}, "
            f"best_seq={best.sequence[:20]}... L={best.L_real:.4f}"
        )
