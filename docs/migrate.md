# RSI-Loop 换机器：一条命令跑起来

`rsi-loop` 是 pip 包（只装代码 236K），S13 数据（236M）单独分发。
目标机器三步：

## 第一步：装代码

```bash
pip install -r requirements.txt    # numpy/pandas/scipy/pyarrow
pip install torch                  # CPU 版；GPU 用 CUDA 版见下
pip install rsi-loop               # 或 pip install -e /path/to/Loop
```

- CPU 版 torch：`pip install torch --index-url https://download.pytorch.org/whl/cpu`
- GPU 版 torch：`pip install torch --index-url https://download.pytorch.org/whl/cu128`

## 第二步：拷数据包

把 `rsipack/`（或含 X/ Y/ train_from_xy.py 的目录）拷到目标机器任意路径。

## 第三步：体检 + 跑（一条命令）

```bash
rsi-alpha --doctor /path/to/rsipack     # 体检：sha1 对账 + 依赖 + 数据口径 + CUDA
rsi-alpha /path/to/rsipack              # 离线 TPE 搜索，默认参数直接跑
```

## 常用参数

```bash
rsi-alpha /path/to/rsipack --iterations 8 --blocks 2 --fixed-epochs 10   # 小规模验证
rsi-alpha /path/to/rsipack --blocks null --fixed-epochs 15               # GPU 全量 123 块
rsi-alpha /path/to/rsipack --online                                     # DeepSeek LLM 当 Refiner
```

## 说明

- 默认离线 TPE 搜索，零 API、零 config 文件。
- GPU 机器自动走 CUDA（`evaluate_recipe` 里自动检测），无需改任何配置。
- 数据包路径可传 `rsipack` 本身（内置 s13/ 子目录），也可传含 X/Y 的目录。
- 产出去 `runs_alpha/<run_id>/`：`summary.json`（最佳配方）、`history.json`（每轮指标）。
