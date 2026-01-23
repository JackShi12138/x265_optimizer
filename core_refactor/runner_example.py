import pandas as pd
import time
import os
import sys

# 适配模块路径 (防止 import 报错)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_refactor.params import SearchSpace
from core_refactor.cost import X265CostEvaluator
from core_refactor.algorithms import RelevancyGuidedOptimizer

# === 配置 ===
EXCEL_FILE_PATH = "标准测试序列码率范围.xlsx"
VIDEO_BASE_PATH = "/home/shiyushen/x265_sequence"
X265_EXEC_PATH = "/home/shiyushen/program/x265_4.0/x265_release"
# 结果输出路径，请按需修改
RESULT_BASE_PATH = "/home/shiyushen/result/offline_optimizer/"


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


def main():
    start_time = time.time()
    print("=== Starting Optimization Process (Refactored & Debugged) ===")

    if not os.path.exists(RESULT_BASE_PATH):
        print(f"Creating output directory: {RESULT_BASE_PATH}")
        os.makedirs(RESULT_BASE_PATH, exist_ok=True)

    video_data = load_video_sequences(EXCEL_FILE_PATH)
    param_space = SearchSpace()
    evaluator = X265CostEvaluator(base_path=RESULT_BASE_PATH, x265_path=X265_EXEC_PATH)

    # 3. 定义优化顺序 (Task 1 Result & PrioritySorter Logic)
    module_order = ["psyrdo", "qcomp", "psyrdoq", "vaq", "cutree"]

    optimizer = RelevancyGuidedOptimizer(
        evaluator=evaluator, param_space=param_space, module_order=module_order
    )

    final_results = {}
    qualities = ["Very Low", "Low", "Medium", "High"]

    for quality in qualities:
        sequences = video_data.get(quality, {})
        if not sequences:
            print(f"\n[Skipping] Target Quality: {quality} (No sequences)")
            continue

        print(f"\n\n{'='*60}")
        print(f">>> Optimizing for Target Quality: {quality} <<<")
        print(f"    Number of sequences: {len(sequences)}")

        # === 关键修正: 强制重置与调试打印 ===
        evaluator.reset()
        evaluator.set_context(quality=quality, module="Init", iteration=0)

        param_space.reset()  # 显式调用新增加的 reset()

        # DEBUG: 打印重置后的参数，确保是默认值
        # 默认值参考：VAQ(str=1.0, mode=2), CUTree(str=2.0), PsyRD(2.0), PsyRDOQ(1.0), QComp(0.6)
        print(f"[DEBUG] Params Reset Check for {quality}:")
        print(param_space.get_all_config())
        # ==================================

        best_config = optimizer.optimize(sequences)
        final_results[quality] = best_config
        print(f">>> Best Config for {quality}: {best_config}")

    end_time = time.time()
    time_str = f"{int(end_time - start_time)} seconds"
    print(f"\nTotal Execution Time: {time_str}")

    output_file = "optimal_params_refactored.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        for quality, params in final_results.items():
            file.write(f"Quality: {quality}\n")
            for m_name, m_params in params.items():
                file.write(f"  Module: {m_name}\n")
                for k, v in m_params.items():
                    file.write(f"    {k}: {v}\n")
            file.write("\n")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()

# nohup python -m core_refactor.runner_example > run.log 2>&1 &
