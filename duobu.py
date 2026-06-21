import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas
from normalization import DynamicNorm
from marina_layers import MARINA

# 固定超参数
SEQ_LEN = 96
# 论文三组预测步长
PRED_LEN_LIST = [24, 48, 168]
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.0002
ALPHA = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集类
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

# 加载数据集
df = pandas.read_csv("./ETT-small/ETTh2.csv")
raw_data = df.iloc[:, 1:].values.astype(np.float32)

# 划分训练/验证集
train_split = int(len(raw_data) * 0.7)
val_split = int(len(raw_data) * 0.15)
train_data = raw_data[:train_split]
val_data = raw_data[train_split:train_split+val_split]

# 归一化：统一使用静态归一化（你实测效果更优）
norm = DynamicNorm(alpha=ALPHA)
norm.fit_train(train_data)
train_data = norm.static_norm(train_data)
val_data = norm.static_norm(val_data)

# 存储所有实验结果
exp_results = []

# 循环遍历不同预测步长
for PRED_LEN in PRED_LEN_LIST:
    print(f"\n========== 开始训练 | 预测步长 = {PRED_LEN} ==========")

    # 构建数据加载器
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

    # 初始化模型
    feat_dim = raw_data.shape[1]
    model = MARINA(
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        feat_dim=feat_dim,
        hidden_dim=200
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 训练循环
    for epoch in range(1, EPOCHS + 1):
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

        # 验证
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

    # 保存当前步长的模型权重
    torch.save(model.state_dict(), f"marina_pred{PRED_LEN}.pth")
    # 记录最终结果
    exp_results.append({
        "pred_len": PRED_LEN,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss
    })

# 汇总打印多步预测结果（直接复制到报告表格）
print("\n" + "="*50)
print("【多步预测实验 最终结果汇总】")
print("="*50)
for res in exp_results:
    print(f"预测步长: {res['pred_len']:3d} | 训练损失: {res['final_train_loss']:.4f} | 验证MSE: {res['final_val_loss']:.4f}")