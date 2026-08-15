"""SNG 架构原型包。

SNG（State-Navigated Generative）是一种新的通用大模型架构：
通用三身份（主控制器 / 附属插件 / Echo-er）+ 池式黑板 + 电源管理
+ 双层仲裁 + 可验证状态（WS 状态读写层 / CRM 可编译规则记忆层 / SVSL
自验证循环头）+ 低比特内存。本包是用于验证机制的最小可运行实现。
"""

from sng.config import Config

__version__ = "0.1.0"
__all__ = ["Config"]
