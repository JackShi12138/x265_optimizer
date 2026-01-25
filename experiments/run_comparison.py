import sys
import os
import time
import pandas as pd
import csv

# 适配路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_refactor.params import SearchSpace
from core_refactor.cost import X265CostEvaluator
from core_refactor.algorithms import RelevancyGuidedOptimizer
from core_refactor.cost import VMAFCostEvaluator
from experiments.baselines.random_search import RandomOptimizer
from experiments.baselines.genetic import GeneticOptimizer

try:
    from experiments.baselines.bayesian import BayesianOptimizer

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# === 配置区域 (请根据服务器实际情况调整) ===
EXCEL_FILE_PATH = "标准测试序列码率范围.xlsx"
VIDEO_BASE_PATH = "/home/shiyushen/x265_sequence"
X265_EXEC_PATH = "/home/shiyushen/program/x265_4.0/x265_release"
# 结果保存路径 (建议与之前的 debug 路径分开)
RESULT_BASE_PATH = "/home/shiyushen/result/comparison_experiment/"

# 基准算法的计算预算 (Evaluations count)
BUDGET = 200


def load_video_sequences(excel_path):
    if not os.path.exists(excel_path):
        print(f"Error: Excel file '{excel_path}' not found.")
        return {}
    excel_file = pd.ExcelFile(excel_path)
    df = excel_file.parse("Sheet1")
    df["Class"] = df["Class"].ffill()
    qualities = ["Very Low", "Low", "Medium", "High"]
    video_sequences_by_quality = {quality: {} for quality in qualities}
    for _, row in df.iterrows():
        video_name = row["Name"]
        video_path = f'{VIDEO_BASE_PATH}/Class{row["Class"]}/{video_name}.yuv'
        for quality in qualities:
            if quality in row:
                video_sequences_by_quality[quality][video_path] = row[quality]
    return video_sequences_by_quality


def run_experiment():
    # 1. 初始化环境
    if not os.path.exists(RESULT_BASE_PATH):
        os.makedirs(RESULT_BASE_PATH)

    # === [修复] 同时清理 CSV 和 TXT 旧文件 ===
    files_to_clean = ["optimization_history.csv", "cost_log_refactor.txt"]

    for filename in files_to_clean:
        if os.path.exists(filename):
            print(f"Removing old log file: {filename}")
            try:
                os.remove(filename)
            except OSError as e:
                print(f"Error removing {filename}: {e}")
    # ========================================

    # 2. 准备数据
    print("Loading video sequences...")
    all_videos = load_video_sequences(EXCEL_FILE_PATH)

    # 3. 初始化评估器 (共享)
    # evaluator = X265CostEvaluator(
    #     base_path=RESULT_BASE_PATH, x265_path=X265_EXEC_PATH, allow_pruning=False
    # )
    evaluator = VMAFCostEvaluator(
        base_path=RESULT_BASE_PATH,
        x265_path=X265_EXEC_PATH,
        vmaf_exec_path="vmaf",
        allow_pruning=False,  # 对比实验建议关闭剪枝
    )

    # 4. 定义要运行的清晰度等级
    # === 修改点：只聚焦 Medium 档位 ===
    target_qualities = ["Medium"]

    for quality in target_qualities:
        sequences = all_videos.get(quality, {})
        if not sequences:
            print(f"Skipping {quality} (No sequences)")
            continue

        print(f"\n{'='*80}")
        print(f"Processing Quality: {quality} (Seqs: {len(sequences)})")
        print(f"{'='*80}")

        # === 算法 1: RG-BCD (Proposed) ===
        print(f"\n>>> Running Algorithm: RG-BCD (Proposed) <<<")
        param_space = SearchSpace()
        evaluator.reset()

        optimizer = RelevancyGuidedOptimizer(evaluator, param_space)
        optimizer.optimize(sequences)

        # === 算法 2: Random Search ===
        print(f"\n>>> Running Algorithm: Random Search <<<")
        param_space = SearchSpace()
        evaluator.reset()

        optimizer = RandomOptimizer(evaluator, param_space, max_evals=BUDGET)
        optimizer.optimize(sequences)

        # === 算法 3: Bayesian Optimization ===
        if OPTUNA_AVAILABLE:
            print(f"\n>>> Running Algorithm: Bayesian Optimization <<<")
            param_space = SearchSpace()
            evaluator.reset()

            optimizer = BayesianOptimizer(evaluator, param_space, max_evals=BUDGET)
            optimizer.optimize(sequences)
        else:
            print(f"\n>>> Bayesian Optimization Skipped (Optuna missing) <<<")

        # === 算法 4: Genetic Algorithm ===
        print(f"\n>>> Running Algorithm: Genetic Algorithm <<<")
        param_space = SearchSpace()
        evaluator.reset()

        # Pop=20, Gen=9 -> Approx 200 evals
        optimizer = GeneticOptimizer(evaluator, param_space, pop_size=20, generations=9)
        optimizer.optimize(sequences)

    print("\nAll experiments finished!")
    print(f"Results saved in: {RESULT_BASE_PATH}")
    print(f"Check 'optimization_history.csv' for data.")


if __name__ == "__main__":
    run_experiment()

# nohup python -m experiments.run_comparison > run.log 2>&1 &
