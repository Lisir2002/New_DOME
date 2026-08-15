"""通用对话插件（chat）：非领域闲聊 / 介绍 / 帮助。

能力 chat：对非游戏指令做通用对话回复（介绍、寒暄、能力说明、兜底引导）。
这是"通用架构 = 下载插件即用"的直观例子：核心骨架不变，
只需注册一个新插件文件即可让系统处理通用对话。
原型为规则/模板实现（正式实现中是独立小模型，≤80MB）。
"""

from __future__ import annotations

from sng.plugins.base import BasePlugin, PluginManifest

_INTRO = (
    "我是 SNG 三身份模型系统：由主控制器统一调度，按需唤醒领域插件，"
    "再由 Echo-er 整理输出。当前搭载了文本游戏、记忆、双语和通用对话四个插件。"
)
_GREET = "你好！我随时可以帮你。你可以下达游戏指令（如“查看”“去厨房”“拾取苹果”“打开大门”），也可以跟我闲聊。"
_HELP = "我能做的：文本游戏操作（查看 / 去房间 / 拾取 / 放下 / 打开大门 / 背包）、状态记忆、中英双语输出，以及通用对话。"
_FAREWELL = "再见，期待下次再聊！"


class ChatPlugin(BasePlugin):
    manifest = PluginManifest(
        plugin_id="chat",
        capabilities=["chat"],
        size_mb=1.5,
        description="通用对话：寒暄 / 介绍 / 能力说明 / 兜底引导",
        judge_rules=["payload 非领域指令时开机"],
        task_types=["generation"],
    )

    def run(self, payload: dict):
        text = (payload.get("text") or "").strip()
        low = text.lower()
        if any(k in text for k in ("你是谁", "介绍", "自我")) or low in ("who are you",):
            reply = _INTRO
        elif any(k in text for k in ("你好", "您好", "嗨", "哈喽", "hello", "hi")) or low in ("hello", "hi", "hey"):
            reply = _GREET
        elif any(k in text for k in ("能做什么", "帮助", "会什么", "功能")) or low in ("help", "what can you do"):
            reply = _HELP
        elif any(k in text for k in ("再见", "拜拜", "bye")) or low in ("bye", "goodbye"):
            reply = _FAREWELL
        else:
            reply = (f"我收到你说的是“{text}”。这句话目前不属于文本游戏指令，"
                     "但我们可以这样聊：你可以问我“你是谁”“你能做什么”，"
                     "或直接下达游戏指令试试。")
        return reply, 0.95, [
            {"type": "chat", "claim": "完成通用对话回复", "precondition": True,
             "check": "utterance"},
        ]
