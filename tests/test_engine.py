"""引擎测试：语言理解（双语指令解析）与状态迁移标注。"""
import pytest

from sng.engine.world import Action, MiniGameEngine


def test_parse_zh_commands():
    e = MiniGameEngine()
    assert e.parse_command("打开大门", "zh") == Action("open", "front_door")
    assert e.parse_command("去厨房", "zh") == Action("go", "kitchen")
    assert e.parse_command("拾取苹果", "zh") == Action("take", "apple")
    assert e.parse_command("放下钥匙", "zh") == Action("drop", "key")
    assert e.parse_command("查看", "zh") == Action("look", "")
    assert e.parse_command("背包", "zh") == Action("inv", "")


def test_parse_en_commands():
    e = MiniGameEngine()
    assert e.parse_command("open the front door", "en") == Action("open", "front_door")
    assert e.parse_command("go to the kitchen", "en") == Action("go", "kitchen")
    assert e.parse_command("pick up the apple", "en") == Action("take", "apple")


def test_parse_unknown_returns_none():
    e = MiniGameEngine()
    assert e.parse_command("随便说点什么", "zh") is None


def test_step_annotations():
    e = MiniGameEngine()
    obs_zh, obs_en, st, info = e.step(Action("go", "garden"))
    assert info["valid"] is True
    assert info["rule_id"] == "move_valid"
    assert info["dispatch"] == {"memory"}
    assert st.location == "garden"
    assert ("location", "set", "garden") in info["slot_ops"]


def test_locked_door_rule():
    e = MiniGameEngine()
    # 玩家初始有钥匙，可打开锁着的门
    _, _, _, info = e.step(Action("open", "front_door"))
    assert info["rule_id"] == "open_locked_with_key"
    assert info["dispatch"] == {"domain"}


def test_observations_bilingual():
    e = MiniGameEngine()
    zh = e.observe("zh")
    en = e.observe("en")
    assert "客厅" in zh
    assert "living room" in en
