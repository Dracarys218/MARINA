import numpy as np
import torch
import pandas
import matplotlib.pyplot as plt
from normalization import DynamicNorm
from marina_layers import MARINA

# 超参（与训练一致）
SEQ_LEN = 96
PRED_LEN = 48
ALPHA = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
df = pandas.read_csv("./ETT-small/ETTh2.csv")
raw_data = df.iloc[:, 1:].values.astype(np.float32)
train_end = int(len(raw_data) * 0.7)
train_data = raw_data[:train_end]
norm = DynamicNorm(alpha=ALPHA)
norm.fit_train(train_data)
all_data = norm.update(raw_data)

# 加载模型
feat_dim = raw_data.shape[1]
model = MARINA(SEQ_LEN, PRED_LEN, feat_dim, hidden_dim=200).to(DEVICE)
model.load_state_dict(torch.load("marina_trained.pth"))
model.eval()

# 计算所有窗口误差
error_list = []
with torch.no_grad():
    for i in range(len(all_data) - SEQ_LEN - PRED_LEN):
        x_np = all_data[i:i+SEQ_LEN]
        y_true_np = all_data[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN]
        
        x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        y_pred = model(x)
        y_pred_np = y_pred.squeeze().cpu().numpy()
        
        err = np.linalg.norm(y_true_np - y_pred_np, ord="fro") / PRED_LEN
        error_list.append(err)

error_arr = np.array(error_list)

# 统计信息
print("===== 误差分布统计 =====")
print(f"最小误差: {np.min(error_arr):.4f}")
print(f"最大误差: {np.max(error_arr):.4f}")
print(f"平均误差: {np.mean(error_arr):.4f}")
print(f"中位数误差: {np.median(error_arr):.4f}")

# 使用90分位数作为阈值（推荐）
THRESHOLD = np.percentile(error_arr, 90)
print(f"\n90%分位数阈值: {THRESHOLD:.4f}")

# 统计异常
anomaly_count = np.sum(error_arr > THRESHOLD)
total = len(error_arr)
print(f"\n===== 异常检测结果 =====")
print(f"总检测窗口数: {total}")
print(f"异常窗口数: {anomaly_count}")
print(f"异常窗口占比: {anomaly_count/total:.2%}")

# 绘图：误差分布直方图
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 中文支持
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(10, 5))
plt.hist(error_arr, bins=50, alpha=0.7, color="skyblue", label="正常窗口误差")
# 绘制阈值线
plt.axvline(x=THRESHOLD, color="red", linestyle="--", linewidth=2, label=f"异常阈值 {THRESHOLD:.4f}")

plt.title("预测误差分布与异常阈值")
plt.xlabel("平均F范数误差")
plt.ylabel("窗口数量")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("error_dist.png", dpi=300, bbox_inches="tight")
plt.show()
print("\n✅ 误差分布图已保存为 error_dist.png")