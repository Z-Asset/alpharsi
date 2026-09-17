#!/usr/bin/env python3
"""目标机体检：对打包目录做 sha1 账物对账 + 依赖检查。

用法（在目标机器上）:
    python migrate_doctor.py --pkg D:/ZR/rsipack

检查项:
  1. manifest.json 里每个文件 sha1 与实际文件逐一对账（账物不符即报 FAIL）
  2. Python 依赖是否可 import（numpy/pandas/scipy/pyarrow/torch）
  3. CUDA 是否可用（决定能不能跑全量 GPU 搜索）
  4. 数据张量形状是否与 S13 口径一致（X=(2833,379,54), Y=(2833,379)）

全部通过才打印 [PASS]，否则 [FAIL] 并退出码 1。
"""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_manifest(pkg: Path) -> tuple[int, int]:
    mf = pkg / "manifest.json"
    if not mf.exists():
        return 0, 0  # 无 manifest，跳过（老包）
    data = json.loads(mf.read_text(encoding="utf-8"))
    ok = fail = 0
    for f in data["files"]:
        p = pkg / f["path"]
        if not p.exists():
            print(f"  [FAIL] 缺失 {f['path']}")
            fail += 1
            continue
        if sha1(p) != f["sha1"]:
            print(f"  [FAIL] sha1 不符 {f['path']}")
            fail += 1
        else:
            ok += 1
    print(f"  sha1 对账: {ok} 通过 / {fail} 失败")
    return ok, fail


def check_deps() -> tuple[int, int]:
    ok = fail = 0
    for mod in ["numpy", "pandas", "scipy", "pyarrow", "torch"]:
        try:
            importlib.import_module(mod)
            ok += 1
        except ImportError:
            print(f"  [FAIL] 缺依赖 {mod}")
            fail += 1
    print(f"  依赖: {ok} 可用 / {fail} 缺失")
    return ok, fail


def check_data(pkg: Path) -> int:
    import numpy as np
    s13 = pkg / "s13"
    x = np.load(s13 / "X" / "X.npy")
    y = np.load(s13 / "Y" / "Y_target.npy")
    if x.shape != (2833, 379, 54):
        print(f"  [FAIL] X 形状 {x.shape} != (2833,379,54)")
        return 1
    if y.shape != (2833, 379):
        print(f"  [FAIL] Y 形状 {y.shape} != (2833,379)")
        return 1
    print(f"  X={x.shape} Y={y.shape} 符合 S13 口径")
    return 0


def check_cuda() -> None:
    import torch
    if torch.cuda.is_available():
        print(f"  CUDA: 可用 ({torch.cuda.get_device_name(0)})，可跑全量 GPU 搜索")
    else:
        print("  CUDA: 不可用（CPU 版 torch 或无卡），只能跑小规模验证")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=".", help="打包目录路径")
    args = ap.parse_args()
    pkg = Path(args.pkg).resolve()

    print(f"[migrate_doctor] 体检 {pkg}")
    ok1, fail1 = check_manifest(pkg)
    ok2, fail2 = check_deps()
    fail3 = check_data(pkg)
    check_cuda()

    total_fail = fail1 + fail2 + fail3
    if total_fail:
        print(f"\n[FAIL] {total_fail} 项未通过")
        sys.exit(1)
    print("\n[PASS] 全部通过，可运行 run_alpha.py")


if __name__ == "__main__":
    main()
