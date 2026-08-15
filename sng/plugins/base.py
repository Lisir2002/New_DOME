"""插件基类与插件清单规范（Manifest）。

每个插件是一个独立文件，携带 manifest（身份卡）：
- capabilities：能力列表（对表 task_type / 仲裁规则）；
- size_mb：内存占用（电源管理依据）；
- judge_rules：自判断器规则（纯规则纠错，成本≈0）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PluginManifest:
    plugin_id: str
    capabilities: list[str]
    size_mb: float
    version: str = "0.1.0"
    description: str = ""
    judge_rules: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)


class BasePlugin:
    """插件基类。子类实现 run()。"""

    manifest: PluginManifest

    def run(self, payload: dict[str, Any]) -> tuple[str, float, list[dict]]:
        """执行插件。返回 (内容, 置信度, 断言列表)。"""
        raise NotImplementedError
