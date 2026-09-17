#!/usr/bin/env python3
"""RSI 框架 + S13 alpha 包 迁移打包器。

把「框架 + 数据 + 训练脚本」打成一个自包含目录，拷到目标机器即可跑。

用法:
    python migrate_pack.py --pkg D:/ZR/Loop/S13_csi500seq_trans --out D:/ZR/rsipack

产出 <out>/ 结构:
    rsi/                   框架（纯 stdlib）
    run_alpha.py           入口
    config/alpha.json      配置
    s13/                   S13 数据 + 训练脚本（去掉冗余的 X_pre.npy）
    requirements.txt       依赖
    MIGRATE.md             目标机三步指引

打包时对每个数据文件算 sha1 写进 manifest.json，目标机 doctor 对账。
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_and_hash(src: Path, dst: Path, root: Path, manifest: list):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append({"path": str(dst.relative_to(root)),
                     "sha1": sha1(dst), "bytes": dst.stat().st_size})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="D:/ZR/Loop/S13_csi500seq_trans",
                    help="S13 数据包路径")
    ap.add_argument("--out", default="D:/ZR/rsipack", help="打包输出目录")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    pkg = Path(args.pkg).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    manifest = []

    # 1) 框架（纯代码）
    shutil.copytree(here / "alpharsi", out / "alpharsi")
    copy_and_hash(here / "run_alpha.py", out / "run_alpha.py", out, manifest)
    copy_and_hash(here / "migrate_doctor.py", out / "migrate_doctor.py", out, manifest)
    shutil.copytree(here / "docs", out / "docs")
    copy_and_hash(here / "requirements.txt", out / "requirements.txt", out, manifest)

    # config：改写 pkg 为相对路径 s13，目标机通用。
    shutil.copytree(here / "config", out / "config")
    cfg_path = out / "config" / "alpha.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["pkg"] = "s13"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    copy_and_hash(here / "config" / "alpha.json", out / "config" / "alpha.json", out, manifest)

    # 2) S13 数据 + 训练脚本（X_pre.npy 冗余，跳过，省 232M）
    s13 = out / "s13"
    for name in ["X.npy", "codes.txt", "dates.txt", "feature_cols.txt"]:
        copy_and_hash(pkg / "X" / name, s13 / "X" / name, out, manifest)
    copy_and_hash(pkg / "Y" / "Y_target.npy", s13 / "Y" / "Y_target.npy", out, manifest)
    copy_and_hash(pkg / "Y" / "Y_meta.json", s13 / "Y" / "Y_meta.json", out, manifest)
    copy_and_hash(pkg / "train_from_xy.py", s13 / "train_from_xy.py", out, manifest)

    # 3) manifest + 指引
    (out / "manifest.json").write_text(
        json.dumps({"files": manifest}, indent=1), encoding="utf-8")

    (out / "MIGRATE.md").write_text(
        "# 迁移三步\n\n"
        "```bash\n"
        "pip install -r requirements.txt   # torch 按需装 CPU 或 CUDA 版\n"
        "python migrate_doctor.py          # sha1 对账，全绿再跑\n"
        "python run_alpha.py config/alpha.json --offline   # 离线 TPE 搜索\n"
        "```\n\n"
        "GPU 全量跑：改 config/alpha.json 的 blocks=null、fixed_epochs=15，"
        "见 docs/alpha_gpu.md。\n",
        encoding="utf-8")

    total = sum(f["bytes"] for f in manifest)
    print(f"[migrate_pack] 打包完成 -> {out}")
    print(f"  文件数 {len(manifest)}，总 {total/1e6:.1f} MB（已省掉 X_pre.npy 232MB）")


if __name__ == "__main__":
    main()
