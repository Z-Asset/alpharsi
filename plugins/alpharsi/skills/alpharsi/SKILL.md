---
name: alpharsi
description: >
  AlphaRSI：三角色自改进闭环，把 alpha 序列模型（S13 等中证 500 复刻包）的训练配方当进化对象。
  Operator 跑 walk-forward、Evaluator 用 OOS_RankIC 验收、Refiner 用 TPE 贝叶斯搜索提议下一组超参。
  核心是离线、可复算、可迁移：pip 装 alpharsi 后一条命令跑，数据包用 sha1 对账分发，GPU 自动走 CUDA。
  当哥说"进化这个 alpha 模型的配方"、"给 S13 扫超参"、"跑 alpharsi"、"把这个包迁移到 GPU 机器"、
  "看看搜索结果找到更优配方没有"时触发。
---

# AlphaRSI

三角色闭环，进化 alpha 模型的训练配方。**不要凭记忆复述参数**——先看
`docs/migrate.md` 与 `docs/alpha_gpu.md`，命令行的东西以 `alpharsi --help` 为准。

## 三个角色

| 角色 | 实现 |
|---|---|
| Operator 运行模型 | `train_from_xy.py` 的 `walk_forward()`（确定性训练脚本） |
| Evaluator 验收模型 | `summarize()` 的 OOS_RankIC / ICIR / SR（封闭） |
| Refiner 改进模型 | 离线 TPE 贝叶斯搜索；在线 DeepSeek LLM |

## 一条命令

```
alpharsi <数据包路径>                # 离线 TPE 搜索，默认参数
alpharsi --doctor <数据包路径>        # 体检：sha1 对账 + 依赖 + 数据口径 + CUDA
alpharsi <路径> --online             # DeepSeek LLM 当 Refiner（需 DEEPSEEK_API_KEY）
```

## 成本约束（必读）

CPU 上真实评估（epochs=10, blocks=2）单点约 345 秒，全量 123 块不可行。
**真正出结论在 GPU 上跑全量**：`--blocks` 不传（None=全量）+ `--fixed-epochs 15`，
单块约 15 秒。详见 `docs/alpha_gpu.md`。

## 硬边界

- **`fixed_epochs` 必须设**：否则 TPE 搜的 epochs 维度在评估时不生效，形成假维度。
- **不改训练函数体**：`walk_forward` / `summarize` 只 import 调用，不改源。
- 配方必须过 `validate_recipe` 合法空间校验，越界拒绝。
- 数据包（X/Y 张量）不进代码仓库，用 `migrate_pack.py` + sha1 对账分发。
- 离线是主推路径（TPE 无 API 依赖）；在线只是 Refiner 换成 LLM。
