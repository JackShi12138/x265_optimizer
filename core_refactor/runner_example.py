import pandas as pd
import time
import os
from .params import SearchSpace
from .cost import X265CostEvaluator
from .algorithms import RelevancyGuidedOptimizer

# --- 配置部分 (迁移自 run.py) ---
# 请根据实际环境调整路径
EXCEL_FILE_PATH = "标准测试序列码率范围.xlsx"
# 假设的视频序列根目录，请按需修改
VIDEO_BASE_PATH = "/home/shiyushen/x265_sequence"
# 假设的x265路径，请按需修改
X265_EXEC_PATH = "/home/shiyushen/program/x265_4.0/x265_release"
# 结果输出路径，请按需修改
RESULT_BASE_PATH = "/home/shiyushen/result/"


def load_video_sequences(excel_path):
    """
    读取Excel文件并按清晰度(Quality)分组构建视频序列字典。
    返回结构: { "Very Low": {path: bitrate, ...}, "Low": {...}, ... }
    """
    if not os.path.exists(excel_path):
        print(
            f"Warning: Excel file '{excel_path}' not found. Using mock data or skipping."
        )
        return {}

    excel_file = pd.ExcelFile(excel_path)
    df = excel_file.parse("Sheet1")
    # 填充合并单元格的 Class 列
    df["Class"] = df["Class"].ffill()

    qualities = ["Very Low", "Low", "Medium", "High"]
    video_sequences_by_quality = {quality: {} for quality in qualities}

    for _, row in df.iterrows():
        video_name = row["Name"]
        # 保持原有的路径拼接逻辑: /ClassX/Name.yuv
        video_path = f'{VIDEO_BASE_PATH}/Class{row["Class"]}/{video_name}.yuv'

        for quality in qualities:
            if quality in row:
                video_sequences_by_quality[quality][video_path] = row[quality]
            else:
                pass
                # print(f"Warning: Quality '{quality}' not found for video {video_name}")

    return video_sequences_by_quality


def main():
    start_time = time.time()
    print("=== Starting Optimization Process (Refactored) ===")

    # 1. 准备数据
    video_data = load_video_sequences(EXCEL_FILE_PATH)
    if not video_data:
        print("No video data loaded or file not found. Please check paths.")
        # 这里不直接退出，以免在无数据环境下无法测试代码逻辑，实际运行时应确保数据存在

    # 2. 初始化核心组件
    # SearchSpace 内部已按照论文设定了所有模块的参数范围和默认值
    param_space = SearchSpace()

    # 初始化成本计算器
    evaluator = X265CostEvaluator(base_path=RESULT_BASE_PATH, x265_path=X265_EXEC_PATH)

    # 3. 定义优化顺序 (Task 1 Result & PrioritySorter Logic)
    # 根据 run.py 中的 priorities:
    #   psyrdo(5) > psyrdoq(4) > qcomp(3) > vaq(2) > cutree(1)
    # PrioritySorter 按优先级降序排列，因此执行顺序如下：
    module_order = ["psyrdo", "psyrdoq", "qcomp", "vaq", "cutree"]

    # 初始化优化器
    # 使用新的 RG-BCD 策略 (Relevancy-Guided Block Coordinate Descent)
    optimizer = RelevancyGuidedOptimizer(
        evaluator=evaluator, param_space=param_space, module_order=module_order
    )

    # 4. 执行优化循环 (针对每个清晰度分别优化)
    final_results = {}

    # 按照 run.py 的逻辑，遍历所有清晰度
    qualities = ["Very Low", "Low", "Medium", "High"]

    for quality in qualities:
        sequences = video_data.get(quality, {})
        if not sequences:
            print(f"\n[Skipping] Target Quality: {quality} (No sequences found)")
            continue

        print(f"\n\n>>> Optimizing for Target Quality: {quality} <<<")
        print(f"    Number of sequences: {len(sequences)}")

        # 重置评估器状态 (清除缓存，重置全局最小Cost，以防不同清晰度间干扰)
        evaluator.reset()

        # 重置参数空间到默认值
        # (论文逻辑：针对每个码率点从初始状态或特定预设开始搜索)
        param_space._init_default_space()

        # --- 核心调用 ---
        best_config = optimizer.optimize(sequences)

        final_results[quality] = best_config
        print(f">>> Best Config for {quality}: {best_config}")

    # 5. 统计与保存
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    time_str = f"{int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds"
    print(f"\nTotal Execution Time: {time_str}")

    # 保存结果 (保持原有格式)
    output_file = "optimal_params_refactored.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        for quality, params in final_results.items():
            file.write(f"Quality: {quality}\n")
            for module_name, module_params in params.items():
                file.write(f"  Module: {module_name}\n")
                for param, value in module_params.items():
                    file.write(f"    {param}: {value}\n")
            file.write("\n")

        file.write("Program Execution Time:\n" + f"  {time_str}")

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
