import numpy as np
import torch
import pandas
import matplotlib.pyplot as plt
from normalization import DynamicNorm
from marina_layers import MARINA

# 超参数（和训练保持一致）
SEQ_LEN = 96
PRED_LEN = 48
ALPHA = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载数据 & 归一化
df = pandas.read_csv("./ETT-small/ETTh2.csv")
raw_data = df.iloc[:, 1:].values.astype(np.float32)

# 复用训练集统计量做归一化
train_end = int(len(raw_data) * 0.7)
train_data = raw_data[:train_end]
norm = DynamicNorm(alpha=ALPHA)
norm.fit_train(train_data)
all_data = norm.update(raw_data)

# 2. 加载模型权重
feat_dim = raw_data.shape[1]
model = MARINA(SEQ_LEN, PRED_LEN, feat_dim, hidden_dim=200).to(DEVICE)
model.load_state_dict(torch.load("marina_trained.pth"))
model.eval()

# 3. 选取一段测试数据（取中间片段）
start_idx = train_end + 200  # 验证集区间
x_np = all_data[start_idx : start_idx + SEQ_LEN]   # 输入序列
y_true_np = all_data[start_idx + SEQ_LEN : start_idx + SEQ_LEN + PRED_LEN]  # 真实值

# 转为模型输入
x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
x = x.unsqueeze(0)  # 增加batch维度

# 4. 模型预测
with torch.no_grad():
    y_pred = model(x)
y_pred_np = y_pred.squeeze(0).cpu().numpy()

# 5. 可视化（选取第1个特征维度绘图）
feat_idx = 0
plt.figure(figsize=(14, 6))

# 绘制历史输入序列
plt.plot(range(SEQ_LEN), x_np[:, feat_idx], label="History Data", color="#1f77b4")
# 绘制真实预测段
plt.plot(range(SEQ_LEN, SEQ_LEN + PRED_LEN), y_true_np[:, feat_idx], label="True Value", color="#2ca02c")
# 绘制模型预测段
plt.plot(range(SEQ_LEN, SEQ_LEN + PRED_LEN), y_pred_np[:, feat_idx], label="Pred Value", color="#d62728", linestyle="--")

plt.title("MARINA 时序预测结果 (ETTh2)", fontsize=12)
plt.xlabel("Time Step")
plt.ylabel("Indicator Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("forecast_result.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ 预测图像已保存为 forecast_result.png")