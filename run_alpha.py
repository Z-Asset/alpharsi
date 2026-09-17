#!/usr/bin/env python3
"""Alpha Model-RSI 闭环入口。

用法:
    python run_alpha.py config/alpha.json --offline   # 离线，确定性配方生成器
    python run_alpha.py config/alpha.json             # 真实跑，DeepSeek LLM 当 Refiner

闭环：Refiner(LLM) 提议训练配方 → 训练脚本 walk-forward → OOS_RankIC 验收
→ 只保留严格更优。快速评估默认 fast_epochs=1 压到秒级，最后全量验证最佳配方。
"""
import sys

from alpharsi.alpha_rsi import run_alpha_loop


def main(argv: list[str]) -> int:
    args = [a for a in argv if a != "--offline"]
    offline = "--offline" in argv
    config = args[0] if args else "config/alpha.json"
    run_alpha_loop(config, offline=offline)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
