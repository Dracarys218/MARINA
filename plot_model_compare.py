import matplotlib.pyplot as plt

# 设置中文支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 数据
models = ["MARINA", "LSTM", "MLP"]
val_mse = [0.3804, 0.4621, 0.5236]
colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]

plt.figure(figsize=(8, 6))
bars = plt.bar(models, val_mse, color=colors, width=0.6)

# 在柱子上标注数值
for bar, val in zip(bars, val_mse):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.4f}", ha="center", va="bottom", fontsize=12)

plt.title("不同模型预测性能对比 (预测步长=48)", fontsize=14)
plt.ylabel("验证集 MSE", fontsize=12)
plt.ylim(0, 0.6)  # 调整y轴范围，让数值标签更清晰
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("model_compare_fixed.png", dpi=300, bbox_inches="tight")
plt.show()