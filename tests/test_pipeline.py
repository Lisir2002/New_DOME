"""端到端管道测试：跑通一回合完整链路。"""
import torch

from sng.config import Config
from sng.pipeline import SNGPipeline
from sng.training import train


def _make_pipe():
    cfg = Config()
    model, _ = train(cfg, epochs=30, n_rollouts=30, seed=7)
    model.eval()
    return cfg, SNGPipeline(cfg, model)


def test_full_turn_chain():
    cfg, pipe = _make_pipe()
    out = pipe.turn("打开大门")
    assert "error" not in out
    assert out["rule"] in ("open_locked_with_key", "open_locked_no_key", "invalid_action")
    assert out["answer"]
    # 状态已提交：门应为打开状态
    assert out["verify_ok"] is True
    # 内存预算约束生效
    assert out["peak_mb"] <= cfg.memory_hard_mb


def test_multi_turn_state_persistence():
    _, pipe = _make_pipe()
    pipe.turn("打开大门")
    pipe.turn("背包")
    out = pipe.turn("查看")
    assert out["answer"]
    assert out["verify_ok"] is True


def test_unparseable_input_routes_to_general_chat():
    _, pipe = _make_pipe()
    out = pipe.turn("完全无关的内容xyz")
    assert out["mode"] == "chat"
    assert out["answer"]
    assert "【回复】" in out["answer"]


def test_general_chat_intents():
    _, pipe = _make_pipe()
    intro = pipe.turn("你是谁")
    assert "SNG" in intro["answer"]
    greet = pipe.turn("你好")
    assert "你好" in greet["answer"]
    # 游戏与闲聊在同一会话内互不影响
    game = pipe.turn("去厨房")
    assert game.get("mode") != "chat"
    assert game["answer"]
