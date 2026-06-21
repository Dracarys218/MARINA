import matplotlib.pyplot as plt

# 1. 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 2. 数据
pod_names = ["carts-tht6g", "carts-4bwfj", "carts-db"]
scores = [0.26, 0.58, 0.44]
colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]
threshold = 0.5

# 3. 画图
plt.figure(figsize=(10, 7))
bars = plt.bar(pod_names, scores, color=colors, width=0.7)

# 标注柱子上的数值
for bar, score in zip(bars, scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{score:.2f}",
        ha="center", va="bottom", fontsize=14
    )

# 异常阈值线
plt.axhline(y=threshold, color="red", linestyle="--", linewidth=2, label=f"异常判定阈值 ({threshold})")

# 标题和标签（现在是中文，不会乱码）
plt.title("各Pod MARINA综合异常分数", fontsize=16)
plt.ylabel("异常分数", fontsize=14)
plt.ylim(0, 0.7)
plt.legend(loc="upper right", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("marina_anomaly_scores_clean.png", dpi=300, bbox_inches="tight")
plt.show()