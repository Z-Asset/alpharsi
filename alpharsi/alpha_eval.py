"""Alpha 模型评估器 —— 把 S13_csi500seq_trans 的训练链路接进三角色闭环。

这一层是 Model-RSI 的 Operator + Evaluator（确定性代码，不是 LLM）：

  Operator  = train_from_xy.walk_forward()   跑 expanding walk-forward
  Evaluator = train_from_xy.summarize()      算 OOS_RankIC / OOS_ICIR / OOS_SR

Refiner（LLM）不在这里；它只负责提议「训练配方」，配方作为 grid 喂给
walk_forward（其内部对 grid 每个候选做验证段选优）。本模块只做两件事：

  1. 把一组合法配方临时塞进 GRIDS[KEY_FOCUS]，跑 walk_forward，算指标。
  2. 把结果整理成 Refiner 能读的、可比较的 LearningSignal。

关键设计：walk_forward 的 grid 来自模块级 GRIDS['trans']，所以评估前要
临时替换、评估后恢复，绝不留脏状态。配方必须过 validate_recipe 校验
（约束到合法空间），不合法直接拒绝——Refiner 的提议永远进不了训练。

可单独命令行验证训练链路（不启动闭环）：
    python -m alpharsi.alpha_eval --pkg D:/ZR/Loop/S13_csi500seq_trans --blocks 1 --max-train 4000
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 非 TTY 下 stdout 行缓冲。
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# 配方合法空间（trans 模型）。Refiner 的提议必须落在这些约束内。
# ---------------------------------------------------------------------------

RECIPE_SPACE: dict[str, Any] = {
    "d_model": [16, 32, 48, 64, 96, 128],
    "nhead": [2, 4, 8],
    "layers": [1, 2, 3],
    "ffn": [64, 128, 256, 512],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "epochs": [3, 5, 10, 15],
    "lr": [0.0003, 0.001, 0.003],
    "batch": [1024, 2048, 4096],
}

# 默认基线：与 GRIDS['trans'] 现有候选一致。
DEFAULT_RECIPE = {"d_model": 32, "nhead": 4, "layers": 1, "ffn": 128,
                  "dropout": 0.1, "epochs": 15, "lr": 0.001, "batch": 4096}

REQUIRED_KEYS = ["d_model", "nhead", "layers", "ffn", "dropout", "epochs", "lr", "batch"]


def validate_recipe(recipe: dict) -> dict:
    """校验并补全一个配方。不合法抛 ValueError（调用方拒绝该提议）。"""
    if not isinstance(recipe, dict):
        raise ValueError("recipe must be an object")
    out: dict = {}
    for k in REQUIRED_KEYS:
        if k not in recipe:
            raise ValueError(f"recipe missing key {k!r}")
        v = recipe[k]
        if k in ("epochs", "batch"):
            v = int(v)
        if k in ("d_model", "ffn"):
            v = int(v)
        if k in ("nhead", "layers"):
            v = int(v)
        if k == "dropout":
            v = float(v)
        if k == "lr":
            v = float(v)
        out[k] = v
    # 结构约束
    if out["d_model"] % out["nhead"] != 0:
        raise ValueError(f"d_model({out['d_model']}) must be divisible by nhead({out['nhead']})")
    for k, v in out.items():
        if k in RECIPE_SPACE and isinstance(RECIPE_SPACE[k], list) and v not in RECIPE_SPACE[k]:
            raise ValueError(f"{k}={v} not in allowed space {RECIPE_SPACE[k]}")
    return out


def _load_module(pkg: Path):
    """从包路径动态加载 train_from_xy.py，不污染 sys.path。"""
    entry = pkg / "train_from_xy.py"
    if not entry.exists():
        raise FileNotFoundError(f"找不到训练入口 {entry}")
    spec = importlib.util.spec_from_file_location("s13_train_from_xy", entry)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {entry}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["s13_train_from_xy"] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class EvalResult:
    """一次配方集合评估的归一化结果。"""
    recipe: dict
    oos_sr: float | None
    oos_rank_ic: float | None
    oos_icir: float | None
    oos_r2: float | None
    n_days: int
    n_blocks: int
    elapsed_s: float
    degenerate: bool = False
    raw: dict = field(default_factory=dict)

    def primary_metric(self) -> float:
        """主验收指标：OOS_RankIC（纯信号质量），NaN 记为 -inf。"""
        if self.oos_rank_ic is None:
            return float("-inf")
        return float(self.oos_rank_ic)

    def to_jsonable(self) -> dict:
        return {
            "recipe": self.recipe,
            "oos_sr": self.oos_sr,
            "oos_rank_ic": self.oos_rank_ic,
            "oos_icir": self.oos_icir,
            "oos_r2": self.oos_r2,
            "n_days": self.n_days,
            "n_blocks": self.n_blocks,
            "elapsed_s": self.elapsed_s,
            "degenerate": self.degenerate,
            "raw": self.raw,
        }

    @classmethod
    def from_jsonable(cls, d: dict) -> "EvalResult":
        return cls(
            recipe=d["recipe"], oos_sr=d["oos_sr"], oos_rank_ic=d["oos_rank_ic"],
            oos_icir=d["oos_icir"], oos_r2=d["oos_r2"], n_days=d["n_days"],
            n_blocks=d["n_blocks"], elapsed_s=d["elapsed_s"],
            degenerate=d.get("degenerate", False), raw=d.get("raw", {}),
        )


# ---------------------------------------------------------------------------
# 评估入口
# ---------------------------------------------------------------------------

def evaluate_recipe(
    pkg: Path,
    recipe: dict,
    *,
    blocks: int = 1,
    max_train: int = 4000,
    L: int = 60,
    seed: int = 42,
    fast_epochs: int | None = None,
    threads: int = 2,
    verbose: bool = False,
) -> EvalResult:
    """评估单个配方：塞进 grid、跑 walk-forward、算 OOS 指标。

    fast_epochs 非 None 时，用该值覆盖配方里的 epochs（快速评估专用），
    让一轮试错从分钟级压到秒级；全量验证时传 None 用配方本身的 epochs。
    """
    import torch
    torch.set_num_threads(threads)

    mod = _load_module(pkg)
    recipe = validate_recipe(recipe)
    if fast_epochs is not None:
        recipe = {**recipe, "epochs": int(fast_epochs)}

    panel, _ = mod._read_pkg_data(pkg, verbose=False)
    X = mod.preprocess_panel(mod.raw_features(panel), mod.PreprocessConfig(), verbose=False)
    R = mod.build_labels(mod.labels(panel), verbose=False)

    # 临时替换 grid，跑完恢复（walk_forward 内部读 GRIDS[KEY_FOCUS]）。
    key = mod.KEY_FOCUS
    saved_grid = mod.GRIDS[key]
    mod.GRIDS[key] = [recipe]
    # 自动检测 CUDA：GPU 机器上走 cuda，CPU 上回退 cpu。无需手改。
    mod.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tmp = Path(__import__("tempfile").mkdtemp(prefix="s13rsi_"))
    t0 = time.time()
    try:
        r = mod.walk_forward(
            X, R, panel.dates, panel.codes, L=L, seed=seed,
            max_train=max_train, out_dir=tmp, max_blocks=blocks, verbose=verbose,
        )
        Rdf = mod.to_frame(R, panel)
        row = mod.summarize(r["pred"], Rdf, r["bench"], name=mod.MODEL_NAME,
                            pred_train=r["pred_train"])
    finally:
        mod.GRIDS[key] = saved_grid
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    def _f(v):
        return None if v is None or (isinstance(v, float) and v != v) else float(v)

    return EvalResult(
        recipe=recipe,
        oos_sr=_f(row.get("OOS_SR")),
        oos_rank_ic=_f(row.get("OOS_RankIC")),
        oos_icir=_f(row.get("OOS_ICIR")),
        oos_r2=_f(row.get("OOS_R2")),
        n_days=int(row.get("n_days", 0)),
        n_blocks=len(r["selected"]),
        elapsed_s=round(time.time() - t0, 1),
        degenerate=bool(row.get("坍缩")),
        raw={k: row.get(k) for k in ("OOS_SR", "OOS_IR", "OOS_RankIC", "OOS_ICIR",
                                     "OOS_R2", "OOS_命中率", "OOS_年化收益", "OOS_最大回撤")},
    )


def evaluate_recipe_isolated(pkg: Path, recipe: dict, *, blocks: int = 1,
                             max_train: int = 4000, L: int = 60, seed: int = 42,
                             fast_epochs: int | None = None, threads: int = 2,
                             timeout_s: float | None = None) -> EvalResult:
    """在独立子进程里评估，隔离 segfault / 内存累积。

    torch 2.13 CPU 版在 Windows + Python 3.14 下多线程不稳定，单轮评估可能
    segfault（exit 139）或超时。子进程隔离后，崩溃只杀掉 worker，主循环
    捕获并继续下一轮，不中断整个闭环。做法与 Optuna 处理不稳定 objective 一致。
    """
    import subprocess

    payload = json.dumps({
        "pkg": str(pkg), "recipe": recipe, "blocks": blocks, "max_train": max_train,
        "L": L, "seed": seed, "fast_epochs": fast_epochs, "threads": threads,
    })
    cmd = [sys.executable, "-m", "alpharsi.alpha_eval", "--worker", "--pkg", str(pkg)]
    proc = subprocess.run(
        cmd, input=payload, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(
            f"worker 崩溃 rc={proc.returncode}（隔离已捕获，跳过本轮）: {' | '.join(tail)}")
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return EvalResult.from_jsonable(json.loads(line))
            except (json.JSONDecodeError, KeyError):
                continue
    raise RuntimeError("worker 无有效结果输出")


def _worker_main():
    """子进程入口：读 stdin 的 JSON 参数，跑评估，stdout 最后一行输出结果 JSON。"""
    import argparse

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--pkg", default=None)
    args, _ = ap.parse_known_args()
    payload = json.loads(sys.stdin.read())
    pkg = Path(payload["pkg"])
    res = evaluate_recipe(
        pkg, payload["recipe"], blocks=int(payload.get("blocks", 1)),
        max_train=int(payload.get("max_train", 4000)), L=int(payload.get("L", 60)),
        seed=int(payload.get("seed", 42)),
        fast_epochs=payload.get("fast_epochs"), threads=int(payload.get("threads", 2)),
        verbose=False,
    )
    print(json.dumps(res.to_jsonable(), ensure_ascii=False))


if __name__ == "__main__":
    import argparse

    if "--worker" in sys.argv:
        _worker_main()
        sys.exit(0)

    ap = argparse.ArgumentParser(description="Alpha 配方评估器（单独验证训练链路）")
    ap.add_argument("--pkg", default="D:/ZR/Loop/S13_csi500seq_trans")
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--max-train", type=int, default=4000)
    ap.add_argument("--fast-epochs", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--recipe", default=None, help="JSON 文件或内联 JSON，缺省用 DEFAULT_RECIPE")
    args = ap.parse_args()

    recipe = DEFAULT_RECIPE
    if args.recipe:
        txt = args.recipe
        if Path(args.recipe).exists():
            txt = Path(args.recipe).read_text(encoding="utf-8")
        recipe = json.loads(txt)

    res = evaluate_recipe(Path(args.pkg), recipe, blocks=args.blocks,
                          max_train=args.max_train, fast_epochs=args.fast_epochs,
                          threads=args.threads, verbose=True)
    print(json.dumps({
        "recipe": res.recipe,
        "OOS_RankIC": res.oos_rank_ic,
        "OOS_ICIR": res.oos_icir,
        "OOS_SR": res.oos_sr,
        "OOS_R2": res.oos_r2,
        "n_days": res.n_days,
        "n_blocks": res.n_blocks,
        "elapsed_s": res.elapsed_s,
        "degenerate": res.degenerate,
    }, ensure_ascii=False, indent=2))
