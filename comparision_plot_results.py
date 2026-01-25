import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# === 配置 ===
CSV_FILE = "optimization_history_vmaf.csv"
OUTPUT_DIR = "experiment_plots"  # [修改] 图片保存目录

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
# 设置 PDF 字体类型，方便论文排版 (Type 42 = TrueType)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def load_and_process_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found.")
        return None

    # 1. 标记算法来源
    def identify_algorithm(module):
        if module == "Random":
            return "Random Search"
        elif module == "BayesOpt":
            return "Bayesian Optimization"
        elif module in ["GA_Init", "GA_Evol"]:
            return "Genetic Algorithm"
        else:
            return "RG-BCD (Proposed)"

    df["Algorithm"] = df["Module"].apply(identify_algorithm)

    # 2. 处理 Eval 计数
    df["Eval_Count"] = df.groupby("Algorithm").cumcount() + 1

    # 3. 计算归一化 Cost (Normalized Cost)
    initial_costs = df.groupby("Algorithm")["Cost"].transform("first")
    df["Norm_Cost"] = df["Cost"] / initial_costs

    # 4. 计算当前最优 (CumMin)
    df["Best_Cost_So_Far"] = df.groupby("Algorithm")["Cost"].cummin()
    df["Norm_Best_Cost"] = df["Best_Cost_So_Far"] / initial_costs

    return df


def plot_convergence(df):
    plt.figure(figsize=(10, 6))

    palette = {
        "RG-BCD (Proposed)": "#d62728",  # 红色
        "Bayesian Optimization": "#1f77b4",  # 蓝色
        "Genetic Algorithm": "#2ca02c",  # 绿色
        "Random Search": "#7f7f7f",  # 灰色
    }

    sns.lineplot(
        data=df,
        x="Eval_Count",
        y="Norm_Best_Cost",
        hue="Algorithm",
        palette=palette,
        linewidth=2.5,
    )

    plt.title(
        "Optimization Convergence (Normalized Cost)", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Number of Evaluations", fontsize=12)
    plt.ylabel("Normalized Cost (Relative to Default)", fontsize=12)

    plt.legend(title="Algorithm", fontsize=10, title_fontsize=11)
    plt.tight_layout()

    # === [修改] 保存为 PNG 和 PDF ===
    save_path_png = os.path.join(OUTPUT_DIR, "convergence_cost.png")
    save_path_pdf = os.path.join(OUTPUT_DIR, "convergence_cost.pdf")

    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_path_pdf, format="pdf", bbox_inches="tight")

    print(f"Saved convergence plots to {OUTPUT_DIR}/")
    # plt.show() # 如果在服务器运行，建议注释掉 show


def plot_vmaf_bitrate(df):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    palette = {
        "RG-BCD (Proposed)": "#d62728",
        "Bayesian Optimization": "#1f77b4",
        "Genetic Algorithm": "#2ca02c",
        "Random Search": "#7f7f7f",
    }

    # --- 1. VMAF 曲线 ---
    sns.lineplot(
        data=df,
        x="Eval_Count",
        y="VMAF",
        hue="Algorithm",
        palette=palette,
        ax=axes[0],
        alpha=0.5,
        linewidth=1.5,
    )
    # 高亮 RG-BCD
    rg_data = df[df["Algorithm"] == "RG-BCD (Proposed)"]
    axes[0].plot(rg_data["Eval_Count"], rg_data["VMAF"], color="#d62728", linewidth=2.5)

    axes[0].set_title("VMAF Score Evolution", fontsize=14)
    axes[0].set_ylabel("VMAF Score", fontsize=12)
    axes[0].legend(loc="lower right")

    # --- 2. Bitrate 曲线 ---
    sns.lineplot(
        data=df,
        x="Eval_Count",
        y="Bitrate",
        hue="Algorithm",
        palette=palette,
        ax=axes[1],
        alpha=0.5,
        linewidth=1.5,
    )
    axes[1].plot(
        rg_data["Eval_Count"], rg_data["Bitrate"], color="#d62728", linewidth=2.5
    )

    axes[1].set_title("Bitrate Evolution", fontsize=14)
    axes[1].set_ylabel("Bitrate (kbps)", fontsize=12)
    axes[1].set_xlabel("Number of Evaluations", fontsize=12)

    plt.tight_layout()

    # === [修改] 保存为 PNG 和 PDF ===
    save_path_png = os.path.join(OUTPUT_DIR, "vmaf_bitrate_evolution.png")
    save_path_pdf = os.path.join(OUTPUT_DIR, "vmaf_bitrate_evolution.pdf")

    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_path_pdf, format="pdf", bbox_inches="tight")

    print(f"Saved VMAF/Bitrate plots to {OUTPUT_DIR}/")
    # plt.show()


def main():
    print(f"Processing {CSV_FILE}...")
    df = load_and_process_data(CSV_FILE)
    if df is not None:
        # 简单打印统计
        print("\n=== Best Normalized Cost Reached ===")
        print(df.groupby("Algorithm")["Norm_Best_Cost"].min().sort_values())

        plot_convergence(df)
        plot_vmaf_bitrate(df)
        print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()
