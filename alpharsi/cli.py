"""alpharsi 命令行入口 —— 一条命令运行 Alpha Model-RSI 闭环。

用法（装包后）:
    alpharsi S13数据包路径            # 离线 TPE 搜索，默认参数
    alpharsi S13数据包路径 --iterations 8 --blocks 2 --fixed-epochs 10
    alpharsi S13数据包路径 --online   # DeepSeek LLM 当 Refiner（需 DEEPSEEK_API_KEY）
    alpharsi --doctor S13数据包路径    # 只体检数据+依赖，不跑

默认离线：不依赖 API、不依赖 config 文件。GPU 机器上自动走 CUDA。
"""
import argparse
import sys

from .alpha_rsi import run_alpha
from .migrate import doctor


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="alpharsi",
        description="Alpha 模型训练配方自进化（Model-RSI，三角色闭环）")
    ap.add_argument("pkg", nargs="?", default=".", help="S13 数据包路径（含 X/Y/train_from_xy.py）")
    ap.add_argument("--doctor", action="store_true", help="只体检数据+依赖，不跑闭环")
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=None, help="walk-forward 块数，None=全量 123 块")
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--fixed-epochs", type=int, default=None, help="固定 epochs，移出搜索空间")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--timeout-s", type=int, default=None, help="单轮评估超时秒数")
    ap.add_argument("--out-dir", default="runs_alpha")
    ap.add_argument("--no-isolation", action="store_true", help="禁用子进程隔离（调试用）")
    ap.add_argument("--online", action="store_true", help="用 DeepSeek LLM 当 Refiner（默认离线 TPE）")
    args = ap.parse_args(argv)

    if args.doctor:
        return doctor(args.pkg)

    settings = {
        "iterations": args.iterations,
        "blocks": args.blocks,
        "max_train": args.max_train,
        "fixed_epochs": args.fixed_epochs,
        "threads": args.threads,
        "seed": args.seed,
        "patience": args.patience,
        "timeout_s": args.timeout_s,
        "out_dir": args.out_dir,
        "isolated": not args.no_isolation,
    }
    run_alpha(args.pkg, settings, offline=not args.online)
    return 0


if __name__ == "__main__":
    sys.exit(main())
