"""调度头（Dispatch Head）。

从 h_t + 状态读取 r 生成调度决策：
- task_types[]：多意图列表（有序）；
- complexity：复杂度估计（0~1，决定仲裁阈值 θ）；
- candidate_plugins：候选插件集（通过能力映射表查 task_type）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sng.config import Config

# 任务类型词表（通用词表，领域无关）
TASK_TYPES = [
    "comprehension", "reasoning", "retrieval", "generation", "lingua", "system",
]
TASK_DIM = len(TASK_TYPES)

# 插件词表（当前注册的插件，控制器学习的调度目标）
PLUGIN_IDS = ["domain", "memory", "lingua"]
PLUGIN_DIM = len(PLUGIN_IDS)


class DispatchHead(nn.Module):
    """调度头：从编码器上下文 h_t + 状态读取 r 生成调度决策。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        d = cfg.emb_dim
        self.fuse = nn.Linear(d * 2, d)
        # 多意图：多标签二分类头
        self.task_classifier = nn.Linear(d, TASK_DIM)
        # 插件选择：多标签二分类头（直接监督引擎真值 dispatch）
        self.plugin_classifier = nn.Linear(d, PLUGIN_DIM)
        # 复杂度：回归头
        self.complexity_reg = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor, r: torch.Tensor) -> dict:
        """前向。返回：
        - task_logits: [TASK_DIM] 多标签 logits
        - plugin_logits: [PLUGIN_DIM] 多标签 logits
        - complexity: [1] 复杂度估计
        """
        fused = self.fuse(torch.cat([h, r]))
        task_logits = self.task_classifier(fused)
        plugin_logits = self.plugin_classifier(fused)
        complexity = torch.sigmoid(self.complexity_reg(fused))
        return {"task_logits": task_logits, "plugin_logits": plugin_logits,
                "complexity": complexity}

    def predict(self, h: torch.Tensor, r: torch.Tensor,
                cfg: Config) -> dict:
        """推理用：取阈值二值化。"""
        out = self.forward(h, r)
        probs = torch.sigmoid(out["task_logits"])
        task_types = [TASK_TYPES[i] for i, p in enumerate(probs) if p > 0.4]
        if not task_types:
            task_types = ["comprehension"]
        c = out["complexity"].item()
        # 插件选择：学习头为主 + 任务类型映射兜底
        plugin_probs = torch.sigmoid(out["plugin_logits"])
        candidate = {PLUGIN_IDS[i] for i, p in enumerate(plugin_probs) if p > 0.4}
        for tt in task_types:
            candidate |= cfg.tasktype_plugins.get(tt, set())
        return {
            "task_types": task_types,
            "complexity": c,
            "candidate_plugins": candidate,
            "task_logits": out["task_logits"],
            "plugin_logits": out["plugin_logits"],
        }

    def dispatch_loss(self, task_logits: torch.Tensor,
                      gt_task_types: list[str]) -> torch.Tensor:
        """多标签分类损失。"""
        targets = torch.zeros(TASK_DIM, dtype=torch.float32)
        for tt in gt_task_types:
            if tt in TASK_TYPES:
                targets[TASK_TYPES.index(tt)] = 1.0
        return nn.functional.binary_cross_entropy_with_logits(task_logits, targets)

    def plugin_loss(self, plugin_logits: torch.Tensor,
                    gt_plugins: set[str]) -> torch.Tensor:
        """插件选择损失：直接监督引擎真值 dispatch 集合。"""
        targets = torch.zeros(PLUGIN_DIM, dtype=torch.float32)
        for pid in gt_plugins:
            if pid in PLUGIN_IDS:
                targets[PLUGIN_IDS.index(pid)] = 1.0
        return nn.functional.binary_cross_entropy_with_logits(plugin_logits, targets)