# AlphaRSI — alpha 模型训练配方自进化

把 alpha 序列模型（如 S13 中证 500 复刻包）的训练配方当进化对象。三角色闭环：Operator 跑 walk-forward、Evaluator 用 OOS_RankIC 验收、Refiner 提议下一组超参，只保留严格更优。参照 MetaRSI-v1 的 Model-RSI 算子与 RSI Kernel 权限边界。

## 三个角色

| 角色 | 实现 |
|---|---|
| Operator 运行模型 | `train_from_xy.py` 的 `walk_forward()`（确定性训练脚本，非 LLM） |
| Evaluator 验收模型 | `summarize()` 的 OOS_RankIC / ICIR / SR（封闭） |
| Refiner 改进模型 | 离线 TPE 贝叶斯搜索；在线 DeepSeek LLM |

循环从不重置：评估 → 诊断 → 提议配方 → 校验 → 评估 → 只保留严格更优 → 继续。

## 一条命令

```bash
pip install -e .                      # 装包（提供 alpharsi 命令）
alpharsi <S13数据包路径>               # 离线 TPE 搜索，默认参数
alpharsi --doctor <S13数据包路径>       # 体检：sha1 对账 + 依赖 + 数据口径 + CUDA
alpharsi <路径> --online              # DeepSeek LLM 当 Refiner（需 DEEPSEEK_API_KEY）
```

默认离线：不依赖 API、不依赖 config 文件。GPU 机器上自动走 CUDA。

## 离线 Refiner 是真智能，不是写死轮询

`alpharsi/offline_refiner.py` 实现 TPE（Tree-structured Parzen Estimator，Optuna 默认算法），在离散超参空间上用历史分位数建模，选 `l(x)/g(x)` 最大的未试点，平衡探索与利用；配合拉丁超立方冷启动、去重、早停。在"超参搜索"这个任务上，通常比让 LLM 盲猜更可靠。

## 成本约束（必读）

CPU 上真实评估（epochs=10, blocks=2）单点约 345 秒，全量 123 块约 12 小时不可行。所以：

- CPU 上做**小规模验证**（`--blocks 2 --fixed-epochs 10`），确认闭环 + TPE 能收敛。
- **最终结论在 GPU 上跑全量**（`--blocks` 不传 = 全量 123 块 + `--fixed-epochs 15`），单块约 15 秒。详见 `docs/alpha_gpu.md`。

配方合法空间在 `alpharsi/alpha_eval.py` 的 `RECIPE_SPACE`：d_model / nhead / layers / ffn / dropout / epochs / lr / batch。

**关键约束**：`fixed_epochs` 必须设。否则 TPE 搜的 epochs 维度在评估时不生效，形成"假维度"，搜索方向全错。

## 目录结构

```
Loop/                      # 本项目 = alpharsi
  alpharsi/
    cli.py                一条命令入口
    alpha_eval.py         Operator + Evaluator（含子进程崩溃隔离）
    alpha_rsi.py          主循环
    offline_refiner.py    TPE 贝叶斯搜索（离线智能 Refiner）
    models.py             ChatModel（仅在线 Refiner 用）
    migrate.py            体检（sha1 对账 + 依赖 + 数据 + CUDA）
  src/index.ts            DSH 插件壳（cordis skill provider）
  plugins/alpharsi/       SKILL.md（Claude Code 插件）
  config/alpha.json       示例配置
  docs/                   migrate.md / alpha_gpu.md
  migrate_pack.py         数据包打包器
  migrate_doctor.py       数据包体检（顶层版）
```

## 换机器

代码是 pip 包（`pip install`），数据（X/Y 张量）单独用 sha1 对账分发。详见 `docs/migrate.md`。

## 硬边界

- **不改训练函数体**：`walk_forward` / `summarize` 只 import 调用，不改源。
- 配方必须过 `validate_recipe` 合法空间校验，越界拒绝。
- 数据包不进代码仓库；S13 只是实验对象，不随框架走。
- Evaluator 封闭：Refiner 看不到、改不了评判器和测试集。

## 相关项目

- `harnessrsi`（独立项目）：通用 Harness-RSI 闭环，进化的是 harness（系统提示词/记忆/技能）。本项目的姐妹项目，走 MetaRSI-v1 的 Harness-RSI 算子。
- `attest`（独立项目）：训练鉴证，与本项目无关。
