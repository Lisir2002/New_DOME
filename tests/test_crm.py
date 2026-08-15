"""CRM 可微索引 + DFA 编译测试。"""
from sng.config import Config
from sng.core.crm import CRM
from sng.engine.world import Action, MiniGameEngine


def test_dfa_matches_engine_for_all_valid_actions():
    cfg = Config()
    e = MiniGameEngine()
    dfa = CRM.build_dfa(e)
    # 遍历若干状态-动作，DFA 结果必须与引擎一致
    e.reset()
    for a in e.valid_actions():
        rule_dfa = CRM.dfa_lookup(dfa, e, a, e.state)
        rule_gt = e._classify(a, e.state)[0]
        assert rule_dfa == rule_gt, f"{a}: dfa={rule_dfa} engine={rule_gt}"
    # 打开锁着的门
    e.step(Action("open", "front_door"))
    for a in e.valid_actions():
        rule_dfa = CRM.dfa_lookup(dfa, e, a, e.state)
        rule_gt = e._classify(a, e.state)[0]
        assert rule_dfa == rule_gt


def test_dfa_is_deterministic_lookup():
    cfg = Config()
    e = MiniGameEngine()
    dfa = CRM.build_dfa(e)
    rule1 = CRM.dfa_lookup(dfa, e, Action("go", "kitchen"), e.state)
    rule2 = CRM.dfa_lookup(dfa, e, Action("go", "kitchen"), e.state)
    assert rule1 == rule2 == "move_valid"


def test_differentiable_index_trains():
    import torch
    cfg = Config()
    e = MiniGameEngine()
    crm = CRM(cfg)
    opt = torch.optim.Adam(crm.parameters(), lr=1e-2)
    samples = []
    e.reset()
    for a in e.valid_actions():
        samples.append((a, e.rule_predicates(a, e.state), e._classify(a, e.state)[0]))
    for _ in range(300):
        opt.zero_grad()
        loss = torch.tensor(0.0)
        for a, preds, rid in samples:
            alpha = crm.forward(a.verb, a.target, preds)
            loss = loss + crm.match_loss(alpha, rid)
        loss.backward()
        opt.step()
    ok = 0
    for a, preds, rid in samples:
        pred, _ = crm.predict_rule(a.verb, a.target, preds)
        ok += pred == rid
    assert ok / len(samples) > 0.9, f"CRM 索引准确率 {ok/len(samples):.2f}"
