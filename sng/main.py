"""SNG 原型 CLI 演示：训练 + 交互。

用法：
  python -m sng.main --train    训练控制器（首次运行）
  python -m sng.main            交互式玩文本游戏（中英双语）
"""

from __future__ import annotations

import argparse
import os

import torch

from sng.config import Config
from sng.core.controller import Controller
from sng.engine.vocab import Vocabulary
from sng.pipeline import SNGPipeline
from sng.training import build_vocab, train

MODEL_PATH = "/workspace/sng_controller.pt"


def load_or_train(cfg: Config) -> Controller:
    if os.path.exists(MODEL_PATH):
        engine = __import__("sng.engine.world", fromlist=["MiniGameEngine"]).MiniGameEngine()
        vocab = build_vocab(cfg, engine)
        model = Controller(cfg, vocab)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        return model
    model, _ = train(cfg)
    torch.save(model.state_dict(), MODEL_PATH)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="SNG 原型演示")
    ap.add_argument("--train", action="store_true", help="重新训练控制器")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    args = ap.parse_args()

    cfg = Config()
    if args.train:
        model, _ = train(cfg)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"训练完成，模型已保存：{MODEL_PATH}")
    else:
        model = load_or_train(cfg)

    pipe = SNGPipeline(cfg, model, lang=args.lang)
    print("SNG 原型已就绪。输入指令（如 查看 / 去厨房 / 拾取苹果 / 打开大门），"
          "输入 quit 退出，输入 help 查看帮助。")
    while True:
        try:
            text = input(f"[{args.lang}]> ").strip()
        except EOFError:
            break
        if text in ("quit", "exit", "退出"):
            break
        if text == "help":
            print("可用指令：查看 / 去{房间} / 拾取{物品} / 放下{物品} / 打开大门 / 背包 / 帮助")
            continue
        out = pipe.turn(text)
        if "error" in out:
            print(out["error"])
            continue
        print(out["answer"])
        print(f"  · 动作={out['action'].verb} 规则={out['rule']} "
              f"验证={'通过' if out['verify_ok'] else '失败'} "
              f"峰值内存={out['peak_mb']:.1f}MB")


if __name__ == "__main__":
    main()
