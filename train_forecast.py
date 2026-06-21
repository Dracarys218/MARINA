import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas
from normalization import DynamicNorm
from marina_layers import MARINA

超参数（和论文保持一致）
SEQ_LEN = 96
PRED_LEN = 48
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


# 加载 ETTh2 数据集
df = pandas.read_csv("./ETT-small/ETTh2.csv")
# 剔除时间列，只保留7个特征
raw_data = df.iloc[:, 1:].values.astype(np.float32)

# 划分训练/验证集
train_ratio = 0.7
val_ratio = 0.15
train_end = int(len(raw_data) * train_ratio)
val_end = train_end + int(len(raw_data) * val_ratio)

train_data = raw_data[:train_end]
val_data = raw_data[train_end:val_end]

# 论文动态归一化
norm = DynamicNorm(alpha=ALPHA)
norm.fit_train(train_data)
# 原动态归一化
# train_data = norm.update(train_data)
# val_data = norm.update(val_data)

# 构建 DataLoader（输出 shape: [32, 96, 7] / [32, 48, 7]）
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

# 损失函数 & 优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 训练主循环
if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        # 训练阶段
        model.train()
        train_loss_total = 0.0
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()

        # 验证阶段
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss_total += loss.item()

        # 计算平均损失并打印
        avg_train = train_loss_total / len(train_loader)
        avg_val = val_loss_total / len(val_loader)
        print(f"Epoch {epoch:2d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    # 保存训练好的模型
    torch.save(model.state_dict(), "marina_trained.pth")
    print("✅ 训练完成，模型已保存为 marina_trained.pth")