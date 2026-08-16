"""记忆插件专项训练测试：词表编码 / 追踪器学习轨迹 / 插件接入与回退。"""

from __future__ import annotations

import random

import pytest
import torch

from sng.engine.world import Action, ITEMS, ROOM_ZH, ROOMS, MiniGameEngine
from sng.plugins.memory import MemoryPlugin
from sng.plugins.memory_model import build_action_vocab, encode_history, train_memory_tracker


@pytest.fixture(scope="module")
def tracker():
    """共享一个已训练的记忆追踪器（加大训练量，背包 ~100% 保证测试稳定）。"""
    engine = MiniGameEngine(seed=7)
    return train_memory_tracker(engine, n_rollouts=300, max_steps=8,
                                epochs=45, hidden=48, verbose=False, seed=7)


def _tok():
    vt, tt, n = build_action_vocab()
    return lambda a: vt[a.verb] * n + tt[a.target] + 1


def _random_history(engine: MiniGameEngine, n: int = 3):
    """随机合法轨迹 -> (历史, 引擎真值状态)。"""
    rng = random.Random(11)
    engine.reset()
    hist = []
    for _ in range(n):
        a = rng.choice(engine.valid_actions())
        _, _, _, info = engine.step(a)
        if info["valid"]:
            hist.append(a)
    return hist, engine.state


def test_action_vocab_and_encode():
    tok = _tok()
    assert tok(Action("go", "kitchen")) > 0
    seq = encode_history([Action("go", "kitchen"), Action("look")], tok)
    assert seq.dim() == 1  # 一维，批量时由训练循环 stack
    assert seq.numel() == 32
    assert seq.dtype == torch.long


def test_tracker_predicts_state_from_history(tracker):
    """追踪器只凭动作轨迹回忆状态，与引擎真值一致。"""
    tok = _tok()
    engine = MiniGameEngine(seed=13)
    hist, st = _random_history(engine, n=4)
    with torch.no_grad():
        loc_l, inv_l, do_l, dl_l = tracker(encode_history(hist, tok))
    assert ROOMS[loc_l.argmax().item()] == st.location
    inv_pred = [ITEMS[i] for i, v in enumerate((torch.sigmoid(inv_l) > 0.5)[0]) if bool(v)]
    assert set(inv_pred) == set(st.inventory)
    assert bool(torch.sigmoid(do_l)[0].item() > 0.5) == bool(st.flags.get("door_open", False))
    assert bool(torch.sigmoid(dl_l)[0].item() > 0.5) == bool(st.flags.get("door_locked", True))


def test_plugin_recalls_with_model(tracker):
    """接入训练模型后：记忆插件用轨迹回忆（不经当前状态），输出含正确位置。"""
    plugin = MemoryPlugin(tracker=tracker, tok=_tok())
    engine = MiniGameEngine(seed=5)
    hist, st = _random_history(engine, n=3)
    content, conf, claims = plugin.run({"history": list(hist), "state": st})
    assert ROOM_ZH[st.location] in content
    assert conf > 0.9
    assert any(c["check"] == "location" for c in claims)
    assert any(c["check"] == "inventory" for c in claims)


def test_plugin_fallback_without_model():
    """未训练（无 tracker）时回退规则实现，不依赖 history。"""
    engine = MiniGameEngine(seed=2)
    plugin = MemoryPlugin(tracker=None, tok=None)
    content, conf, _ = plugin.run({"state": engine.state})
    assert "客厅" in content
    assert conf > 0.9


def test_long_trajectory_recall(tracker):
    """长轨迹（8 步）后仍能回忆正确状态，验证长叙事记忆。"""
    tok = _tok()
    engine = MiniGameEngine(seed=17)
    hist, st = _random_history(engine, n=8)
    with torch.no_grad():
        loc_l, inv_l, _, _ = tracker(encode_history(hist, tok))
    assert ROOMS[loc_l.argmax().item()] == st.location
    inv_pred = [ITEMS[i] for i, v in enumerate((torch.sigmoid(inv_l) > 0.5)[0]) if bool(v)]
    assert set(inv_pred) == set(st.inventory)
