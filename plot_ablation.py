import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# === 配置 ===
FILES = {
    "Proposed Order (RG-BCD)": "ablation_proposed.csv",
    "Reverse Order": "ablation_reverse.csv",
    "Random Order": "ablation_random.csv",
}
OUTPUT_DIR = "experiment_plots"

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 设置风格
sns.set_theme(style="whitegrid", font_scale=1.3)  # 字体稍微大一点
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42


def load_data():
    dfs = []
    base_cost = None

    # 先读取 Proposed 以确定归一化基准
    if os.path.exists(FILES["Proposed Order (RG-BCD)"]):
        df_prop = pd.read_csv(FILES["Proposed Order (RG-BCD)"])
        base_cost = df_prop["Cost"].iloc[
            0
        ]  # 使用 Proposed 的初始 Cost 作为所有变体的基准
    else:
        print(
            "Warning: Proposed result not found. Normalization might be inconsistent."
        )

    for label, filename in FILES.items():
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found, skipping.")
            continue

        df = pd.read_csv(filename)
        df["Variant"] = label

        # 重建 Eval Count (从 1 开始)
        df["Eval_Count"] = range(1, len(df) + 1)

        # 计算 "截止当前的最优值" (CumMin)
        df["Best_Cost_So_Far"] = df["Cost"].cummin()

        # 归一化
        if base_cost:
            df["Norm_Cost"] = df["Best_Cost_So_Far"] / base_cost
        else:
            df["Norm_Cost"] = df["Best_Cost_So_Far"] / df["Cost"].iloc[0]

        dfs.append(df)

    if not dfs:
        return None

    return pd.concat(dfs)


def plot_ablation_highlight(df):
    plt.figure(figsize=(10, 6))

    # === 1. 颜色与线宽设计 ===
    # Proposed 用鲜艳红色 + 加粗，Reverse 用深蓝，Random 用浅灰
    palette = {
        "Proposed Order (RG-BCD)": "#d62728",
        "Reverse Order": "#1f77b4",
        "Random Order": "#999999",  # 灰色淡化干扰
    }

    # 绘制主曲线
    sns.lineplot(
        data=df,
        x="Eval_Count",
        y="Norm_Cost",
        hue="Variant",
        palette=palette,
        linewidth=2.0,  # 默认线宽
        style="Variant",  # 不同线型
        dashes={
            "Proposed Order (RG-BCD)": (None, None),
            "Reverse Order": (None, None),
            "Random Order": (2, 2),
        },  # Random 用虚线
    )

    # === 2. 突出 Proposed ===
    # 单独重画 Proposed 的线，使其更粗
    proposed_df = df[df["Variant"] == "Proposed Order (RG-BCD)"]
    if not proposed_df.empty:
        plt.plot(
            proposed_df["Eval_Count"],
            proposed_df["Norm_Cost"],
            color="#d62728",
            linewidth=3.5,
            label="_nolegend_",
        )

    # === 3. 添加参考线与标注 ===
    if not proposed_df.empty:
        # 获取 Proposed 的最终收敛值
        final_cost_proposed = proposed_df["Norm_Cost"].iloc[-1]

        # 画一条水平虚线
        plt.axhline(
            y=final_cost_proposed,
            color="#d62728",
            linestyle=":",
            alpha=0.6,
            linewidth=1.5,
        )
        plt.text(
            proposed_df["Eval_Count"].max(),
            final_cost_proposed * 1.0005,
            " Proposed Final Cost",
            color="#d62728",
            va="bottom",
            ha="right",
            fontsize=10,
            fontweight="bold",
        )

        # 标注：找出 Proposed 达到 99.9% 最终性能的点（“肘部”点）
        # 这里的逻辑是：何时 Cost 降到了 (MinCost + 0.001) 以下
        threshold = final_cost_proposed + 0.002
        fast_point = proposed_df[proposed_df["Norm_Cost"] <= threshold]

        if not fast_point.empty:
            x_fast = fast_point["Eval_Count"].iloc[0]
            y_fast = fast_point["Norm_Cost"].iloc[0]

            # 画一个圆点
            plt.plot(x_fast, y_fast, "o", color="#d62728", markersize=8)

            # 添加箭头注释
            plt.annotate(
                f"Fast Convergence\n(~{x_fast} evals)",
                xy=(x_fast, y_fast),
                xytext=(x_fast + 20, y_fast + 0.01),
                arrowprops=dict(facecolor="black", arrowstyle="->", alpha=0.6),
                fontsize=11,
                fontweight="bold",
                color="#333333",
            )

    # === 4. 图表细节美化 ===
    plt.title(
        "Ablation Study: Efficiency of Relevancy-Guided Ordering",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Number of Evaluations", fontsize=12)
    plt.ylabel("Normalized Cost (Relative to Default)", fontsize=12)

    # 限制 X 轴：如果后面拖得很长且平，可以只展示前 100-150 次
    # 根据实际数据调整，例如：plt.xlim(0, 150)
    # plt.xlim(0, 150)

    # 限制 Y 轴：聚焦于下降区域
    # plt.ylim(0.98, 1.01) # 需要根据数据实际范围调整

    plt.legend(title="Optimization Order", loc="upper right", frameon=True, shadow=True)
    plt.tight_layout()

    # 保存
    save_path = os.path.join(OUTPUT_DIR, "ablation_order_highlight")
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_path}.pdf", format="pdf", bbox_inches="tight")
    print(f"Saved highlighted plot to {save_path}.png/.pdf")


if __name__ == "__main__":
    print("Loading ablation data...")
    df = load_data()
    if df is not None:
        plot_ablation_highlight(df)
    else:
        print("No data found. Please run experiments/run_ablation.py first.")
