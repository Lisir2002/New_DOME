"""WS 状态读写层测试：编解码、读取、确定性写入。"""
import torch

from sng.config import Config
from sng.core.ws import StateStream, STATE_VOCAB
from sng.engine.world import Action, GameState, MiniGameEngine


def test_encode_decode_roundtrip():
    cfg = Config()
    ws = StateStream(cfg)
    e = MiniGameEngine()
    st = e.state
    S = ws.encode_state(st)
    back = ws.decode_state(S)
    assert back.location == st.location
    assert set(back.inventory) == set(st.inventory)
    assert back.flags == st.flags


def test_read_shape_and_alpha():
    cfg = Config()
    ws = StateStream(cfg)
    S = torch.zeros(cfg.n_slots, cfg.emb_dim)
    q = torch.randn(cfg.emb_dim)
    r, alpha = ws.read(S, q)
    assert r.shape == (cfg.emb_dim,)
    assert alpha.shape == (cfg.n_slots,)
    assert torch.isclose(alpha.sum(), torch.tensor(1.0))


def test_apply_ops_deterministic():
    cfg = Config()
    ws = StateStream(cfg)
    e = MiniGameEngine()
    S = ws.encode_state(e.state)
    ops = [("location", "set", "garden")]
    S2 = ws.apply_ops(S, ops, lambda s: ws.state_emb(
        torch.tensor(STATE_VOCAB.get(s, 0), dtype=torch.long)))
    assert ws.decode_state(S2).location == "garden"


def test_hybrid_addressing_regions():
    cfg = Config()
    ws = StateStream(cfg)
    S = torch.zeros(cfg.n_slots, cfg.emb_dim)
    h = torch.randn(cfg.emb_dim)
    upd = ws.write_candidate(S, h)
    assert upd["typed_targets"].shape == (cfg.n_typed, cfg.emb_dim)
    assert upd["free_delta"].shape == (cfg.n_slots - cfg.slot_free, cfg.emb_dim)
    assert torch.isclose(upd["free_attn"].sum(), torch.tensor(1.0))
