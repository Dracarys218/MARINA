import torch
import torch.nn as nn

# 基础 MLP 块
class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLPBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# 时序模块
class TemporalModule(nn.Module):
    def __init__(self, feat_dim, hidden_dim=200):
        super(TemporalModule, self).__init__()
        self.mlp1 = MLPBlock(feat_dim, hidden_dim, hidden_dim)
        self.mlp2 = MLPBlock(hidden_dim, hidden_dim, hidden_dim)
        self.mlp3 = MLPBlock(hidden_dim, hidden_dim, feat_dim)

    def forward(self, x):
        out1 = self.mlp1(x)
        out2 = self.mlp2(out1)
        out3 = self.mlp3(out2)
        return x + out3

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, feat_dim):
        super(SpatialAttention, self).__init__()
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_weight = torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1)
        attn_out = torch.matmul(attn_weight, v)
        return self.out_proj(attn_out)

# MARINA 主模型（全程不用解包语法）
class MARINA(nn.Module):
    def __init__(self, seq_len, pred_len, feat_dim, hidden_dim=200):
        super(MARINA, self).__init__()
        self.temporal = TemporalModule(feat_dim, hidden_dim)
        self.spatial_attn = SpatialAttention(feat_dim)
        self.len_proj = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # 时序特征
        x = self.temporal(x)
        # 空间注意力
        x = self.spatial_attn(x)
        # 维度变换  [B, L, F] -> [B, F, L]
        x = x.transpose(1, 2)
        # 序列长度预测
        x = self.len_proj(x)
        # 还原维度 [B, pred_len, F]
        x = x.transpose(1, 2)
        return x