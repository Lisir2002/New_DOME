"""全局配置：维度、内存预算、槽位布局、插件注册等。

本文件集中定义原型用到的所有超参数与常量，方便统一调整。
内存预算严格控制在 50MB 以内（原型为机制验证，实际占用远小于预算）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- 维度 ----
    emb_dim: int = 128            # 词嵌入 / 状态槽维度
    n_slots: int = 128            # WS 状态槽总数
    n_typed: int = 16             # 类型化槽数量（位置 1 + 背包 8 + flag 8）
    n_free: int = 112             # 自由槽数量
    enc_hidden: int = 128         # 编码器（GRU）隐层
    n_rules: int = 16             # CRM 规则条数上限（与引擎规则表对齐）
    max_inv: int = 8              # 背包容量
    n_flags: int = 8              # 布尔 flag 数量

    # ---- 状态槽布局（类型化槽的固定语义映射，schema 决定，无需学习寻址）----
    slot_location: int = 0        # 槽 0：当前位置
    slot_inv: int = 1             # 槽 1..8：背包物品（MAX_INV 个）
    slot_flag: int = 9            # 槽 9..16：布尔 flag（door_open / door_locked ...）
    slot_free: int = 16           # 槽 16..127：自由槽（情节/临时）

    # ---- 机制超参 ----
    svsl_max_iters: int = 3       # SVSL 自验证循环最大迭代 K
    svsl_warm_start: bool = True  # 是否用 CRM DFA 做确定性验证的兜底
    free_slot_temp: float = 1.0   # 自由槽软寻址温度
    crm_sparse_lambda: float = 0.01  # CRM 稀疏正则强度
    crm_dfa_confidence: float = 0.6  # CRM 编译门槛：置信度阈值

    # ---- 内存预算（文档约束：50MB，最多 80MB）----
    memory_budget_mb: int = 50
    memory_hard_mb: int = 80

    # ---- 训练 ----
    lr: float = 3e-3
    epochs: int = 60
    batch_size: int = 32
    seed: int = 42

    # ---- 语言 ----
    languages: list = field(default_factory=lambda: ["zh", "en"])
    default_lang: str = "zh"

    # ---- 插件注册（capability -> 插件 id）----
    capability_plugin: dict = field(default_factory=lambda: {
        "domain_query": "domain",   # 领域规则查询（物品位置 / 门状态）
        "recall": "memory",         # 记忆召回（背包 / 事实）
        "translate": "lingua",      # 双语转换
    })

    # ---- 任务类型 -> 需要的插件（先验映射，兜底用；精确选择由调度头学习）----
    tasktype_plugins: dict = field(default_factory=lambda: {
        "comprehension": {"domain", "memory"},
        "reasoning": {"domain"},
        "retrieval": {"domain", "memory"},
        "generation": set(),
        "lingua": {"lingua"},
        "system": {"memory"},
    })

    def typed_region(self) -> range:
        return range(self.n_typed)

    def free_region(self) -> range:
        return range(self.slot_free, self.n_slots)
