import numpy as np
import torch
from marina_layers import MARINA
from normalization import Dynamic

# 加载训练好的模型 + 测试数据
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 100
PRED_LEN = 1
ANOMALY_THRESHOLD = 0.8  # 异常阈值 γ（根据误差分布调优）

# 加载模型
model = MARINA(SEQ_LEN, PRED_LEN, dim=7).to(DEVICE)
model.load_state_dict(torch.load("./marina_forecast_best.pth"))
model.eval()

# 加载带异常的测试数据（SMAP/SMD 等）
test_data = np.load("anomaly_data/smap_test.npy")
norm = DynamicNorm(alpha=0.1)

# 逐段推理 + 异常判断
anomaly_count = 0
with torch.no_grad():
    for i in range(len(test_data)-SEQ_LEN):
        x = test[i:i+SEQ_LEN].T
        y_true = test[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN].T
        x = torch.tensor(x).unsqueeze(0).to(DEVICE)
        y_true = torch.tensor(y_true).unsqueeze(0).to(DEVICE)

        y_pred = model(x)
        # 计算F范数误差（论文异常分数）
        error = torch.norm(y_pred - y_true, p="fro").item()
        if error > ANOMALY_THRESHOLD:
            anomaly_count += 1
            print(f"检测到异常点，误差：{error:.3f}")

print(f"总计检测异常数量：{anomaly_count}")