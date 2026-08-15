"""端到端链路：控制器 -> 仲裁 -> 插件 -> 验证 -> Echo-er。

每回合：
1. 语言理解（引擎解析指令 -> 结构化动作）；
2. 控制器推理（调度决策 + CRM 规则 + WS 确定性状态更新 + SVSL 验证）；
3. 池式总线仲裁唤醒插件 -> 插件写结论池；
4. SVSL 验证报告；
5. 通知 Echo-er 生成最终回答。
"""

from __future__ import annotations

import torch

from sng.config import Config
from sng.core.controller import Controller
from sng.core.pool import MessageBus, VerificationReport
from sng.core.ws import StateStream
from sng.echoer import Echoer
from sng.engine.vocab import Vocabulary
from sng.engine.world import Action, MiniGameEngine
from sng.plugins.chat import ChatPlugin
from sng.plugins.domain import DomainPlugin
from sng.plugins.lingua import LinguaPlugin
from sng.plugins.memory import MemoryPlugin


class SNGPipeline:
    """SNG 原型端到端链路。"""

    def __init__(self, cfg: Config, controller: Controller,
                 lang: str = "zh") -> None:
        self.cfg = cfg
        self.controller = controller
        self.engine = MiniGameEngine()
        self.bus = MessageBus(cfg)
        self.echoer = Echoer(lang)
        # 插件注册（独立文件，可下载即注册即用）
        self.plugins: dict[str, object] = {}
        for p in (DomainPlugin(), MemoryPlugin(), LinguaPlugin(), ChatPlugin()):
            self.bus.register_plugin(p.manifest.plugin_id, p.manifest.size_mb)
            self.plugins[p.manifest.plugin_id] = p
        self.S = self.controller.ws.encode_state(self.engine.state)
        self.lang = lang

    def reset(self) -> None:
        self.engine.reset()
        self.S = self.controller.ws.encode_state(self.engine.state)
        self.bus.reset()

    # ---- 一回合 ----
    def turn(self, text: str, budget_mb: float | None = None,
             bilingual: bool = False) -> dict:
        a = self.engine.parse_command(text, self.lang)
        if a is None:
            # 非领域指令 -> 通用对话链路（控制器判定域外，交通用插件处理）
            return self._chat_turn(text, bilingual)
        budget = budget_mb or (self.cfg.memory_budget_mb - 10.0)
        obs_ids = self.controller.vocab.encode(
            self.engine.observe(self.lang), self.lang, max_len=48)

        # 控制器推理（调度 + 规则 + 状态更新 + 验证）
        out = self.controller.infer(obs_ids, a, self.S, self.engine,
                                    self.engine.state, use_dfa=True)
        self.S = out["S_final"]
        # 先提交状态到引擎：让插件/观察读到"动作后"的一致状态
        state_before = self.engine.state.clone()
        self._commit_to_engine(a, out["rule_id"])

        # 池式总线：仲裁唤醒插件
        task_id = self.bus.dispatch(
            task_types=out["task_types"], complexity=out["complexity"],
            candidate_plugins=out["candidate_plugins"], budget_mb=budget,
            payload={"text": text, "action": a})
        # 执行被唤醒的插件，写入结论池
        room_zh = self._room_zh(self.engine.state)
        woken = [pid for pid, s in self.bus.pm.plugins.items()
                 if s.name == "ON"]
        for pid in woken:
            plugin = self.plugins[pid]
            payload = {"state": self.engine.state, "state_before": state_before,
                       "action": a, "action_zh": self.engine.action_text(a, "zh"),
                       "text": self.engine.observe(self.lang),
                       "room_zh": room_zh,
                       "target_lang": "en" if bilingual else "zh"}
            content, conf, claims = plugin.run(payload)
            self.bus.publish_result(pid, task_id, content, conf, claims)

        # SVSL 验证报告（确定性一致性检查）
        ok, issues = self.svsl_verify()
        report = VerificationReport(task_id, ok, issues)
        if not ok:
            self.S = self._rollback()
        # 通知 Echo-er
        self.bus.request_generation(task_id, report)
        req = self.bus.gen_requests[-1]
        answer = self.echoer.answer(req, bilingual=bilingual)

        return {
            "action": a, "rule": out["rule_id"], "ops": out["ops"],
            "dispatch": {"task_types": out["task_types"],
                         "plugins": out["candidate_plugins"]},
            "woken": woken, "verify_ok": ok, "verify_issues": issues,
            "svsl_iters": out["svsl_iters"], "verify_score": out["verify_score"],
            "peak_mb": self.bus.pm.peak_memory(),
            "answer": answer, "obs": self.engine.observe(self.lang),
        }

    # ---- 通用对话链路：非领域指令 -> 指令池 -> chat 插件 -> 结论池 -> Echo-er ----
    def _chat_turn(self, text: str, bilingual: bool = False) -> dict:
        budget = self.cfg.memory_budget_mb - 10.0
        task_id = self.bus.dispatch(
            task_types=["generation"], complexity=0.2,
            candidate_plugins={"chat"}, budget_mb=budget,
            payload={"text": text})
        woken = [pid for pid, s in self.bus.pm.plugins.items()
                 if s.name == "ON"]
        for pid in woken:
            plugin = self.plugins[pid]
            content, conf, claims = plugin.run(
                {"text": text, "target_lang": "en" if bilingual else "zh"})
            self.bus.publish_result(pid, task_id, content, conf, claims)
        ok, issues = self.svsl_verify()
        report = VerificationReport(task_id, ok, issues)
        self.bus.request_generation(task_id, report)
        req = self.bus.gen_requests[-1]
        answer = self.echoer.answer(req, bilingual=bilingual)
        return {
            "mode": "chat", "woken": woken, "verify_ok": ok,
            "verify_issues": issues, "peak_mb": self.bus.pm.peak_memory(),
            "answer": answer,
        }

    def svsl_verify(self) -> tuple[bool, list[str]]:
        from sng.core.svsl import SVSL
        return SVSL.consistency_check(self.S, self.cfg)

    def _rollback(self):
        return self.controller.ws.encode_state(self.engine.state)

    def _room_zh(self, st):
        from sng.engine.world import ROOM_ZH
        return ROOM_ZH.get(st.location, st.location)

    def _commit_to_engine(self, a: Action, rule_id: str) -> None:
        if rule_id in ("move_valid", "take_valid", "drop_valid",
                       "open_locked_with_key", "open_unlocked"):
            st = self.engine.state
            ops = self.engine.transition_ops(rule_id, a)
            for key, op, val in ops:
                if key == "location":
                    st.location = str(val)
                elif key == "inv":
                    if op == "append":
                        st.inventory.append(str(val))
                    elif op == "remove":
                        if str(val) in st.inventory:
                            st.inventory.remove(str(val))
                elif key == "room":
                    room_items = st.items_at.setdefault(st.location, [])
                    if op == "remove" and str(val) in room_items:
                        room_items.remove(str(val))
                    elif op == "add" and str(val) not in room_items:
                        room_items.append(str(val))
                elif key.startswith("flag."):
                    st.flags[key.split(".")[1]] = bool(val)
