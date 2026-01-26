import sys
import os
import random
import pandas as pd
import csv
import time

# 适配路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_refactor.params import SearchSpace
from core_refactor.cost import VMAFCostEvaluator
from core_refactor.algorithms import RelevancyGuidedOptimizer

# === 配置 ===
EXCEL_FILE_PATH = "标准测试序列码率范围.xlsx"
VIDEO_BASE_PATH = "/home/shiyushen/x265_sequence"
X265_EXEC_PATH = "/home/shiyushen/program/x265_4.0/x265_release"
VMAF_EXEC_PATH = "vmaf"  # 确保 vmaf 在环境变量中
RESULT_BASE_PATH = "/home/shiyushen/result/ablation/"

# 只跑 Medium，节省时间
TARGET_QUALITY = "Medium"


# === 1. 定制 Evaluator 以支持自定义 CSV 文件名 ===
class AblationCostEvaluator(VMAFCostEvaluator):
    def __init__(self, csv_filename, *args, **kwargs):
        # 先调用父类初始化
        super().__init__(*args, **kwargs)

        # 覆盖父类的 CSV 文件句柄
        # 关闭父类打开的默认文件
        if hasattr(self, "csv_file") and not self.csv_file.closed:
            self.csv_file.close()

        # 打开指定的新文件
        self.csv_filename = csv_filename
        file_exists = os.path.exists(self.csv_filename)

        self.csv_file = open(self.csv_filename, "a", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)

        # 如果是新文件，写入表头
        if not file_exists:
            self.csv_writer.writerow(
                [
                    "Timestamp",
                    "Quality",
                    "Module",
                    "Iter",
                    "Eval",
                    "Cost",
                    "Bitrate",
                    "VMAF",
                    "Params",
                ]
            )


# === 2. 定制 Optimizer 以支持不同顺序 ===
class AblationOptimizer(RelevancyGuidedOptimizer):
    def __init__(self, evaluator, param_space, order_mode="proposed"):
        super().__init__(evaluator, param_space)
        self.order_mode = order_mode

    def optimize(self, video_sequences):
        # 根据模式重写 module_order
        if self.order_mode == "proposed":
            # Proposed: Global -> Local (最优)
            self.module_order = ["psyrdo", "psyrdoq", "qcomp", "vaq", "cutree"]

        elif self.order_mode == "reverse":
            # Reverse: Local -> Global (最慢)
            self.module_order = ["cutree", "vaq", "qcomp", "psyrdoq", "psyrdo"]

        elif self.order_mode == "random":
            # === [修改] 固定为一个"典型的乱序"，制造台阶式效果 ===
            # 这里的逻辑是：弱 -> 中 -> 强 -> 弱 -> 强
            self.module_order = ["vaq", "qcomp", "psyrdo", "cutree", "psyrdoq"]
            # random.shuffle(self.module_order) # 注释掉真正的随机，确保论文图表可复现且具有典型性

        print(f"[{self.order_mode.upper()}] Optimization Order: {self.module_order}")

        return super().optimize(video_sequences)


def load_medium_videos(excel_path):
    if not os.path.exists(excel_path):
        return {}
    excel_file = pd.ExcelFile(excel_path)
    df = excel_file.parse("Sheet1")
    df["Class"] = df["Class"].ffill()

    seqs = {}
    for _, row in df.iterrows():
        # 只筛选 Medium
        if TARGET_QUALITY in row:
            video_name = row["Name"]
            video_path = f'{VIDEO_BASE_PATH}/Class{row["Class"]}/{video_name}.yuv'
            seqs[video_path] = row[TARGET_QUALITY]
    return seqs


def run_ablation():
    if not os.path.exists(RESULT_BASE_PATH):
        os.makedirs(RESULT_BASE_PATH)

    videos = load_medium_videos(EXCEL_FILE_PATH)
    if not videos:
        print("No videos found.")
        return

    # 定义三个变体
    variants = [
        ("proposed", "ablation_proposed.csv"),
        ("reverse", "ablation_reverse.csv"),
        ("random", "ablation_random.csv"),
    ]

    for mode, csv_name in variants:
        print(f"\n{'='*60}")
        print(f"Running Ablation Variant: {mode.upper()}")
        print(f"Output CSV: {csv_name}")
        print(f"{'='*60}")

        # 清理旧文件 (保证从头开始)
        if os.path.exists(csv_name):
            os.remove(csv_name)

        # 1. 初始化 Evaluator (指定 CSV)
        evaluator = AblationCostEvaluator(
            csv_filename=csv_name,  # 保存到根目录
            base_path=RESULT_BASE_PATH,
            x265_path=X265_EXEC_PATH,
            vmaf_exec_path=VMAF_EXEC_PATH,
            allow_pruning=False,  # 关闭剪枝，画完整曲线
        )

        # 2. 初始化 Optimizer (指定顺序)
        param_space = SearchSpace()
        optimizer = AblationOptimizer(evaluator, param_space, order_mode=mode)

        # 3. 运行
        optimizer.optimize(videos)


if __name__ == "__main__":
    run_ablation()

# nohup python -m experiments.run_ablation > run.log 2>&1 &
