import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas

SEQ_LEN = 96
PRED_LEN = 48
BATCH_SIZE = 32

class TimeDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len]
        return x, y

# 加载数据
df = pandas.read_csv("./ETT-small/ETTh2.csv")
raw_data = df.iloc[:, 1:].values.astype(np.float32)
train_arr = raw_data[:int(len(raw_data)*0.7)]

train_loader = DataLoader(TimeDataset(train_arr, SEQ_LEN, PRED_LEN), batch_size=BATCH_SIZE)

# 打印真实维度
for x, y in train_loader:
    print("x 张量形状:", x.shape)
    print("x 维度数:", x.dim())
    break