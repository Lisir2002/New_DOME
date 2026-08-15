"""统一训练循环（§6 阶段 1-3 的原型实现）。

训练控制器端到端：调度头 + WS 世界模型 + CRM 规则索引 + SVSL 验证/修正，
统一损失 = λ1·L_dispatch + λ2·L_comp + λ3·L_crm + λ4·L_state
          + λ5·L_verify + λ6·L_correct + λ7·L_sparse。
"""

from __future__ import annotations

import random

import torch

from sng.config import Config
from sng.core.controller import Controller
from sng.engine.vocab import Vocabulary
from sng.engine.world import MiniGameEngine

LAMBDA = {
    "L_dispatch": 1.0, "L_comp": 0.5, "L_crm": 1.0, "L_plugin": 1.0,
    "L_state": 1.0, "L_verify": 0.5, "L_correct": 0.5, "L_sparse": 0.05,
}


def build_vocab(cfg: Config, engine: MiniGameEngine) -> Vocabulary:
    """用引擎双语观察语料构建固定词表。"""
    vocab = Vocabulary()
    corpus: list[tuple[str, str]] = []
    engine.reset()
    for _ in range(120):
        a = random.choice(engine.valid_actions() +
                          [engine.parse_command(c, "zh") for c in
                           ["查看", "背包", "帮助", "去厨房", "拾取苹果", "打开大门"]
                           if engine.parse_command(c, "zh")])
        if a is None:
            continue
        obs_zh, obs_en, _, _ = engine.step(a)
        corpus.append((obs_zh, "zh"))
        corpus.append((obs_en, "en"))
    vocab.build(corpus)
    return vocab


def make_dataset(cfg: Config, engine: MiniGameEngine, n_rollouts: int = 80,
                 n_steps: int = 6) -> list[dict]:
    engine.reset()
    return [s for _ in range(n_rollouts) for s in engine.rollout(n_steps)]


def train(cfg: Config, epochs: int | None = None,
          n_rollouts: int = 80, seed: int = 0) -> tuple[Controller, dict]:
    random.seed(seed)
    torch.manual_seed(seed)
    engine = MiniGameEngine()
    vocab = build_vocab(cfg, engine)
    model = Controller(cfg, vocab)
    data = make_dataset(cfg, engine, n_rollouts=n_rollouts)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    epochs = epochs or cfg.epochs

    history = {"loss": [], "dispatch_acc": [], "crm_acc": []}
    for ep in range(epochs):
        random.shuffle(data)
        total = {k: 0.0 for k in LAMBDA}
        for sample in data:
            opt.zero_grad()
            losses = model.forward_train(sample, engine)
            loss = sum(LAMBDA[k] * losses[k] for k in LAMBDA)
            loss.backward()
            opt.step()
            for k in LAMBDA:
                total[k] += losses[k].item()
        avg = {k: v / len(data) for k, v in total.items()}
        history["loss"].append(sum(avg.values()))
        if ep % 10 == 0 or ep == epochs - 1:
            accs = evaluate(model, engine, 200)
            history["dispatch_acc"].append(accs["dispatch"])
            history["crm_acc"].append(accs["crm"])
            print(f"[epoch {ep:>3}] loss={sum(avg.values()):.4f} "
                  f"dispatch_acc={accs['dispatch']:.3f} plugin_acc={accs['plugin']:.3f} "
                  f"crm_acc={accs['crm']:.3f} "
                  f"| L_dispatch={avg['L_dispatch']:.3f} L_plugin={avg['L_plugin']:.3f} "
                  f"L_crm={avg['L_crm']:.3f} L_state={avg['L_state']:.4f} "
                  f"L_verify={avg['L_verify']:.4f}")
    return model, history


def evaluate(model: Controller, engine: MiniGameEngine, n: int = 200) -> dict:
    """调度（task_type 与插件选择）与 CRM 规则预测的准确率。"""
    engine.reset()
    samples = engine.rollout(n)
    disp_hit = 0
    plug_hit = 0
    crm_hit = 0
    with torch.no_grad():
        for s in samples:
            obs_ids = model.vocab.encode(s["obs_zh"], "zh", max_len=48)
            h = model.encode_obs(obs_ids)
            S0 = model.ws.encode_state(s["state_before"])
            r, _ = model.ws.read(S0, model.read_q(h))
            disp = model.dispatch.predict(h, r, model.cfg)
            if s["task_type"] in disp["task_types"]:
                disp_hit += 1
            if s["dispatch"] <= disp["candidate_plugins"]:
                plug_hit += 1
            preds = engine.rule_predicates(s["action"], s["state_before"])
            rule, _ = model.crm.predict_rule(s["action"].verb, s["action"].target, preds)
            if rule == s["rule_id"]:
                crm_hit += 1
    return {"dispatch": disp_hit / n, "plugin": plug_hit / n, "crm": crm_hit / n}


if __name__ == "__main__":
    model, hist = train(Config())
    torch.save(model.state_dict(), "/workspace/sng_controller.pt")
    print("已保存模型权重到 /workspace/sng_controller.pt")
