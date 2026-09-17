"""AlphaRSI — alpha 模型训练配方自进化闭环（Model-RSI）。

三个角色：Operator（walk-forward 训练脚本）、Evaluator（OOS_RankIC 验收）、
Refiner（离线 TPE 贝叶斯搜索 / 在线 DeepSeek LLM）。进化对象是训练配方
（d_model / nhead / layers / ffn / dropout / epochs / lr / batch），不动模型权重。

一条命令（装包后）：
    alpharsi <S13数据包路径>            # 离线 TPE 搜索，默认参数
    alpharsi --doctor <S13数据包路径>    # 体检数据+依赖
    alpharsi <路径> --online            # DeepSeek LLM 当 Refiner
"""
from .alpha_rsi import run_alpha, run_alpha_loop, AlphaRunStats
from .alpha_eval import EvalResult, RECIPE_SPACE, evaluate_recipe
from .offline_refiner import OfflineTPERefiner
from .models import ChatModel

__all__ = [
    "run_alpha", "run_alpha_loop", "AlphaRunStats",
    "EvalResult", "RECIPE_SPACE", "evaluate_recipe",
    "OfflineTPERefiner", "ChatModel",
]
