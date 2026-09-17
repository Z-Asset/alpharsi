# Alpha Model-RSI 在 GPU 上全量验证

CPU 上单点真实评估（blocks=2, epochs=10）约 345 秒，全量 123 块不可行。
**真正出结论必须在 GPU 机器上跑**。本页是 GPU 上的完整操作。

## 前置

- 有 CUDA 的机器，装 `torch`（CUDA 版）
- 把整个 `S13_csi500seq_trans` 目录拷过去（含 X/、Y/、train_from_xy.py）
- 把本项目 `Loop/` 的 `rsi/`、`run_alpha.py`、`config/alpha.json` 拷过去

## GPU 全量配置

`config/alpha.json` 的 `settings` 改成：

```json
{
  "pkg": "<GPU机器上的 S13_csi500seq_trans 绝对路径>",
  "model": { "model": "deepseek-chat", "base_url": "https://api.deepseek.com/v1", "api_key": "" },
  "settings": {
    "iterations": 6,
    "blocks": null,          // null = 全量 123 块
    "max_train": 200000,     // 与官方默认一致
    "fast_epochs": null,     // 用配方真实 epochs
    "fixed_epochs": 15,      // 与官方默认一致，固定掉避免假维度
    "threads": 8,
    "L": 60,
    "seed": 42,
    "patience": 3,
    "isolated": true,
    "timeout_s": 7200,       // 单轮上限 2 小时
    "out_dir": "runs_alpha"
  }
}
```

注意 `blocks: null`。`evaluate_recipe` 的 `max_blocks` 参数传 `None` 时，
`walk_forward` 跑全部 123 块。这是唯一的最终口径。

## 跑

```bash
# 离线（TPE 贝叶斯搜索，无需 API）
python run_alpha.py config/alpha.json --offline

# 或在线（DeepSeek LLM 当 Refiner）
export DEEPSEEK_API_KEY=sk-xxxx
python run_alpha.py config/alpha.json
```

## 产出

`runs_alpha/<run_id>/` 里：

- `summary.json` —— 最佳配方 + 最佳 OOS_RankIC
- `history.json` —— 每轮的配方、RankIC/ICIR/SR、接受/拒绝、耗时

拿到最佳配方后，与现有 `model_best.pth`（OOS_SR=0.8556）对比：
手动 `python train_from_xy.py --run --save` 会按官方门禁（OOS_SR 不劣于
base 才覆盖）把新配方落成工件。

## 关键提醒

- **别在 CPU 上跑 blocks=8/epochs=10**：单点 40 分钟，6 轮要 4 小时以上。
- **GPU 上单块 ~15 秒**（官方 README 实测），全量 123 块约 30 分钟，
  6 轮搜索 3 小时左右，可行。
- `fixed_epochs` 必须设：否则 TPE 会搜 epochs，但 `fast_epochs=null` 时
  epochs 又真生效，会出现「搜索维度与评估维度不一致」的假维度问题。
- 设备自动检测：`evaluate_recipe` 里 `mod.DEVICE = "cuda" if torch.cuda.is_available()
  else "cpu"`，GPU 机器上无需任何改动，自动走 CUDA。
