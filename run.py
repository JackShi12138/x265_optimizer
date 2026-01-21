from core import ParameterManager, CostCalculator, PrioritySorter, ParameterOptimizer
import numpy as np
from decimal import Decimal, getcontext
import pandas as pd
import time

start_time = time.time()
# read video sequences and their bitrates
excel_file = pd.ExcelFile("标准测试序列码率范围.xlsx")
df = excel_file.parse("Sheet1")
df["Class"] = df["Class"].ffill()
qualities = ["Very Low", "Low", "Medium", "High"]
video_sequences = {quality: {} for quality in qualities}
for _, row in df.iterrows():
    video_name = row["Name"]
    video_path = f'/home/shiyushen/x265_sequence/Class{row["Class"]}/{video_name}.yuv'
    for quality in qualities:
        video_sequences[quality][video_path] = row[quality]

# set parameter ranges and initial values
getcontext().prec = 2
param_ranges = {
    "vaq": {
        "aq-mode": [0, 1, 2, 3, 4],
        "aq-strength": [float(Decimal(i) / Decimal(10)) for i in range(0, 31)],
    },
    "cutree": {
        "cutree": [0, 1],
        "cutree-strength": [float(Decimal(i) / Decimal(100)) for i in range(0, 26)],
    },
    "psyrdo": {
        "rd": [1, 2, 3, 5],
        "psy-rd": [float(Decimal(i) / Decimal(10)) for i in range(0, 51)],
    },
    "psyrdoq": {
        "rdoq-level": [0, 1, 2],
        "psy-rdoq": [float(Decimal(i) / Decimal(10)) for i in range(0, 101)]
        + [i for i in range(11, 51)],
    },
    "qcomp": {"qcomp": [float(Decimal(i) / Decimal(100)) for i in range(50, 101)]},
}
initial_values = {
    "vaq": {"aq-mode": 2, "aq-strength": 1.0},
    "cutree": {"cutree": 1, "cutree-strength": 2.0},
    "psyrdo": {"rd": 3, "psy-rd": 2.0},
    "psyrdoq": {"rdoq-level": 2, "psy-rdoq": 1.0},
    "qcomp": {"qcomp": 0.6},
}

# create a parameter optimizer
param_manager = ParameterManager(param_ranges, initial_values)
cost_calculator = CostCalculator()
modules = ["vaq", "cutree", "psyrdo", "psyrdoq", "qcomp"]
module_priorities = {"vaq": 2, "cutree": 1, "psyrdo": 5, "psyrdoq": 4, "qcomp": 3}
priority_sorter = PrioritySorter(modules, module_priorities)

# optimize parameters
optimizer = ParameterOptimizer(
    param_manager=param_manager,
    cost_calculator=cost_calculator,
    priority_sorter=priority_sorter,
)
best_params = optimizer.optimize(video_sequences=video_sequences)

end_time = time.time()
elapsed_time = end_time - start_time
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"程序运行时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
print("最优参数组合:", best_params)
# 保存最优参数组合为文本文件
with open("optimal_params.txt", "w", encoding="utf-8") as file:
    for quality, params in best_params.items():
        file.write(f"Quality: {quality}\n")
        for module, module_params in params.items():
            file.write(f"  Module: {module}\n")
            for param, value in module_params.items():
                file.write(f"    {param}: {value}\n")
        file.write("\n")
    
    file.write("Program Execution Time:\n" f"  {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")

