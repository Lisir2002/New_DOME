"""池式消息总线 + 电源管理（Pool-based Message Bus）。

实现"指令池 → 仲裁 → 插件开机 → 结论池 → Echo-er"的域无关黑板架构。
- 插件三态：关机（默认，零占用）/ 唤醒 / 探针；
- 双层仲裁：集中仲裁（主推）+ 插件自判断器（纯规则纠错）；
- 电源管理：内存式子 = 常驻(主模型+仲裁+Echo-er) + Σ_{开机插件}。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PluginState(Enum):
    OFF = 0
    PROBE = 1
    ON = 2


@dataclass
class DispatchRequest:
    """指令池消息：控制器写，仲裁读。"""
    task_id: str
    task_types: list[str]
    complexity: float
    candidate_plugins: set[str]
    budget_mb: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """结论池消息：插件写，Echo-er 读。"""
    task_id: str
    plugin_id: str
    content: str
    confidence: float
    claims: list[dict] = field(default_factory=list)  # 断言列表


@dataclass
class VerificationReport:
    """验证报告：验证器写，通知 Echo-er。"""
    task_id: str
    passed: bool
    issues: list[str] = field(default_factory=list)


@dataclass
class GenerationRequest:
    """Echo-er 请求：由结论池有结果时发。"""
    task_id: str
    results: list[AnalysisResult] = field(default_factory=list)
    verification: VerificationReport | None = None


class PowerManager:
    """电源管理器：跟踪每个插件状态与内存占用。"""

    def __init__(self, constant_mb: float = 10.0) -> None:
        self.constant_mb = constant_mb
        self.plugins: dict[str, PluginState] = {}
        self.plugin_sizes: dict[str, float] = {}

    def register(self, plugin_id: str, size_mb: float) -> None:
        self.plugins[plugin_id] = PluginState.OFF
        self.plugin_sizes[plugin_id] = size_mb

    def set_state(self, plugin_id: str, state: PluginState) -> None:
        if plugin_id in self.plugins:
            self.plugins[plugin_id] = state

    def peak_memory(self) -> float:
        on = sum(self.plugin_sizes[k]
                 for k, s in self.plugins.items() if s == PluginState.ON)
        return self.constant_mb + on

    def __repr__(self) -> str:
        on = [k for k, s in self.plugins.items() if s == PluginState.ON]
        off = [k for k, s in self.plugins.items() if s == PluginState.OFF]
        return f"PM({self.peak_memory():.1f}MB on={on} off={len(off)}pcs)"


class Arbiter:
    """集中仲裁器（双层仲裁先验层）。"""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.complexity_threshold = 0.3

    def decide(self, req: DispatchRequest, pm: PowerManager) -> set[str]:
        """仲裁决策：返回本任务应保持开机的插件集合（与当前状态无关）。

        候选插件按预算筛选。
        """
        candidates = set(req.candidate_plugins) & set(pm.plugins)
        # 预算硬约束：budget_mb 为插件预算（常驻内存已独立计）
        budget = req.budget_mb
        # 候选总占用在预算内则全开
        total = sum(pm.plugin_sizes.get(p, 0) for p in candidates)
        if total <= budget:
            return candidates
        # 按 size 升序，贪心装入
        sorted_p = sorted(candidates, key=lambda p: pm.plugin_sizes.get(p, 0))
        selected: set[str] = set()
        used = 0.0
        for p in sorted_p:
            s = pm.plugin_sizes.get(p, 0)
            if used + s <= budget:
                selected.add(p)
                used += s
        return selected


class PluginJudge:
    """插件自判断器（纯规则纠错，成本≈0）。"""

    @staticmethod
    def confirm(plugin_id: str, req: DispatchRequest) -> bool:
        """插件自判断是否应该开机。"""
        return plugin_id in req.candidate_plugins

    @staticmethod
    def request_wake(plugin_id: str, req: DispatchRequest) -> bool:
        """请求仲裁未派发的开机（纠错漏派）。"""
        return False  # 原型：仅仲裁主导


class MessageBus:
    """池式消息总线，管理指令池与结论池。"""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.dispatch_pool: list[DispatchRequest] = []
        self.conclusion_pool: list[AnalysisResult] = []
        self.gen_requests: list[GenerationRequest] = []
        self.pm = PowerManager(constant_mb=10.0)
        self.arbiter = Arbiter(cfg)
        self._task_counter = 0

    def register_plugin(self, pid: str, size_mb: float) -> None:
        self.pm.register(pid, size_mb)

    def dispatch(self, task_types: list[str], complexity: float,
                 candidate_plugins: set[str], budget_mb: float,
                 payload: dict | None = None) -> str:
        self._task_counter += 1
        tid = f"task_{self._task_counter}"
        req = DispatchRequest(tid, task_types, complexity, candidate_plugins,
                              budget_mb, payload or {})
        self.dispatch_pool.append(req)
        # 仲裁
        to_wake = self.arbiter.decide(req, self.pm)
        # 按需开机、用完即停：关闭本任务不再需要的插件
        for pid in list(self.pm.plugins):
            if self.pm.plugins[pid] == PluginState.ON and pid not in to_wake:
                self.pm.set_state(pid, PluginState.OFF)
        for pid in to_wake:
            self.pm.set_state(pid, PluginState.ON)
        return tid

    def publish_result(self, pid: str, tid: str, content: str,
                       confidence: float, claims: list | None = None) -> None:
        r = AnalysisResult(tid, pid, content, confidence, claims or [])
        self.conclusion_pool.append(r)

    def request_generation(self, tid: str,
                           verification: VerificationReport | None = None) -> None:
        results = [r for r in self.conclusion_pool if r.task_id == tid]
        self.gen_requests.append(GenerationRequest(tid, results, verification))

    def reset(self) -> None:
        self.dispatch_pool.clear()
        self.conclusion_pool.clear()
        self.gen_requests.clear()
        for pid in self.pm.plugins:
            self.pm.set_state(pid, PluginState.OFF)