# train_baseline.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas
from normalization import DynamicNorm
from baseline_models import LSTMForecast, MLPForecast

# ========== 超参数（与MARINA保持完全一致） ==========
SEQ_LEN = 96
PRED_LEN = 48
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.0002
ALPHA = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 数据集类 ==========
class TimeDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = torch.from_numpy(data).float()
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x, y

# ========== 加载与预处理数据 ==========
df = pandas.read_csv("./ETT-small/ETTh2.csv")
raw_data = df.iloc[:, 1:].values.astype(np.float32)

# 划分训练/验证集
train_split = int(len(raw_data) * 0.7)
val_split = int(len(raw_data) * 0.15)
train_data = raw_data[:train_split]
val_data = raw_data[train_split:train_split+val_split]

# 统一使用 静态归一化（和最优配置一致）
norm = DynamicNorm(alpha=ALPHA)
norm.fit_train(train_data)
train_data = norm.static_norm(train_data)
val_data = norm.static_norm(val_data)

# 数据加载器
train_loader = DataLoader(
    TimeDataset(train_data, SEQ_LEN, PRED_LEN),
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    TimeDataset(val_data, SEQ_LEN, PRED_LEN),
    batch_size=BATCH_SIZE,
    shuffle=False
)

feat_dim = raw_data.shape[1]

# ========== 通用训练函数 ==========
def train_model(model, model_name):
    print(f"\n===== 开始训练 {model_name} =====")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), f"{model_name}_pred48.pth")
    return train_loss, val_loss

# ========== 依次训练 LSTM、MLP ==========
if __name__ == "__main__":
    results = {}

    # 1. 训练 LSTM
    lstm_model = LSTMForecast(feat_dim=feat_dim, hidden_dim=64, pred_len=PRED_LEN).to(DEVICE)
    lstm_train_loss, lstm_val_loss = train_model(lstm_model, "LSTM")
    results["LSTM"] = {"train_loss": lstm_train_loss, "val_loss": lstm_val_loss}

    # 2. 训练 MLP
    mlp_model = MLPForecast(seq_len=SEQ_LEN, feat_dim=feat_dim, hidden_dim=200, pred_len=PRED_LEN).to(DEVICE)
    mlp_train_loss, mlp_val_loss = train_model(mlp_model, "MLP")
    results["MLP"] = {"train_loss": mlp_train_loss, "val_loss": mlp_val_loss}

    # ========== 汇总对比结果 ==========
    print("\n" + "="*60)
    print("【基线模型 vs MARINA 最终结果汇总 (PRED_LEN=48)】")
    print("="*60)
    # 填入你之前 MARINA 48步的结果
    marina_train = 0.1278
    marina_val = 0.3804
    print(f"MARINA   | 训练损失: {marina_train:.4f} | 验证MSE: {marina_val:.4f}")
    print(f"LSTM     | 训练损失: {results['LSTM']['train_loss']:.4f} | 验证MSE: {results['LSTM']['val_loss']:.4f}")
    print(f"MLP      | 训练损失: {results['MLP']['train_loss']:.4f} | 验证MSE: {results['MLP']['val_loss']:.4f}")