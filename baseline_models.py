# baseline_models.py
import torch
import torch.nn as nn

# LSTM 基线模型
class LSTMForecast(nn.Module):
    def __init__(self, feat_dim, hidden_dim=64, pred_len=48):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(feat_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, pred_len * feat_dim)
        self.pred_len = pred_len
        self.feat_dim = feat_dim

    def forward(self, x):
        # x: [B, seq_len, feat_dim]
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步输出
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        # 重塑为 [B, pred_len, feat_dim]
        return out.view(-1, self.pred_len, self.feat_dim)

# 纯 MLP 基线模型
class MLPForecast(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_dim=200, pred_len=48):
        super(MLPForecast, self).__init__()
        self.in_dim = seq_len * feat_dim
        self.fc1 = nn.Linear(self.in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, pred_len * feat_dim)
        self.pred_len = pred_len
        self.feat_dim = feat_dim

    def forward(self, x):
        B = x.shape[0]
        # 展平序列维度
        x = x.reshape(B, -1)
        x = torch.relu(self.fc1(x))
        out = self.fc2(x)
        return out.view(-1, self.pred_len, self.feat_dim)