"""池式消息总线 + 电源管理测试。"""
from sng.config import Config
from sng.core.pool import MessageBus, PluginState


def test_power_management_off_by_default():
    cfg = Config()
    bus = MessageBus(cfg)
    bus.register_plugin("domain", 4.0)
    bus.register_plugin("memory", 2.0)
    bus.register_plugin("lingua", 3.0)
    assert bus.pm.peak_memory() == 10.0  # 常驻 10MB，插件全部关机
    assert all(s == PluginState.OFF for s in bus.pm.plugins.values())


def test_arbiter_wakes_and_obeys_budget():
    cfg = Config()
    bus = MessageBus(cfg)
    bus.register_plugin("domain", 4.0)
    bus.register_plugin("memory", 2.0)
    bus.register_plugin("lingua", 3.0)
    # 小预算只允许 memory 开机
    tid = bus.dispatch(["retrieval"], 0.5, {"memory", "domain"}, budget_mb=3.0)
    on = [p for p, s in bus.pm.plugins.items() if s == PluginState.ON]
    assert "memory" in on
    assert "domain" not in on
    assert bus.pm.peak_memory() <= 13.0


def test_dispatch_flows_to_conclusion_pool():
    cfg = Config()
    bus = MessageBus(cfg)
    bus.register_plugin("memory", 2.0)
    tid = bus.dispatch(["retrieval"], 0.2, {"memory"}, budget_mb=10.0)
    bus.publish_result("memory", tid, "背包里有：钥匙", 0.99)
    bus.request_generation(tid)
    req = bus.gen_requests[-1]
    assert req.task_id == tid
    assert len(req.results) == 1
    assert req.results[0].plugin_id == "memory"
