import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class DistillPacket:
    """Universal Knowledge Port 封裝協議"""
    soft_logits: torch.Tensor
    feat_summary: torch.Tensor
    relation_code: torch.Tensor
    attn_summary: Optional[torch.Tensor]
    confidence_score: float

class UDLSLinear(nn.Module):
    """
    基於 W ≈ Decode(C, S, R, A) 的新權重表示層
    Level 1: C (Indices), Level 2: S (Scale), Level 3: R (Residual), Level 4: A (Anchor)
    """
    def __init__(self, in_features, out_features, codebook_size=256, group_size=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        # A: Anchor Codebook (常駐 L3 Cache/Shared Memory)
        self.codebook = nn.Parameter(torch.randn(codebook_size, group_size) * 0.02)
        
        # C: Code Indices (Buffer, 不直接參與梯度更新)
        num_groups = (in_features * out_features) // group_size
        self.register_buffer('C', torch.randint(0, codebook_size, (num_groups,)))
        
        # S: Group Scales (Level 2 更新)
        self.S = nn.Parameter(torch.ones(num_groups))
        
        # R: Sparse Residuals (Level 3 高精度更新)
        self.R = nn.Parameter(torch.zeros(out_features, in_features))
        
        # A: Bias/Anchor statistics
        self.bias = nn.Parameter(torch.zeros(out_features))

    def _decode(self):
        # 核心解碼邏輯：以計算換取頻寬
        w_base = self.codebook[self.C] * self.S.unsqueeze(-1)
        w_base = w_base.view(self.out_features, self.in_features)
        return w_base + self.R

    def forward(self, x):
        return F.linear(x, self._decode(), self.bias)

class UniversalPort(nn.Module):
    """跨架構蒸餾接口：將 Teacher 訊號標準化"""
    def __init__(self, t_dim, s_dim):
        super().__init__()
        self.projector = nn.Linear(t_dim, s_dim)

    def forward(self, t_logits, t_hidden, t_attn=None, conf=1.0):
        feat_summary = self.projector(t_hidden)
        # 計算 Token 間關係矩陣 (Relation Match)
        rel = torch.bmm(t_hidden, t_hidden.transpose(1, 2))
        rel_code = F.softmax(rel / (t_hidden.size(-1)**0.5), dim=-1)
        
        return DistillPacket(t_logits, feat_summary, rel_code, t_attn, conf)

class UDLSTrainer:
    """五維 Loss 動態平衡優化器"""
    def __init__(self, model, alpha=1.0, beta=0.5):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        # 分級優化：不同 Level 設置不同學習率
        self.optimizer = torch.optim.AdamW([
            {'params': [model.S, model.codebook], 'lr': 1e-3}, # Level 1-2
            {'params': [model.R], 'lr': 1e-4}                 # Level 3
        ])

    def compute_loss(self, s_logits, s_feat, packet: DistillPacket):
        # 1. Soft Target Loss
        loss_soft = F.kl_div(
            F.log_softmax(s_logits/2, dim=-1), 
            F.softmax(packet.soft_logits/2, dim=-1), 
            reduction='batchmean'
        ) * 4
        # 2. Feature Align Loss
        loss_feat = F.mse_loss(s_feat, packet.feat_summary)
        # 3. Relation Match Loss
        s_rel = torch.bmm(s_feat, s_feat.transpose(1, 2))
        s_rel_code = F.softmax(s_rel / (s_feat.size(-1)**0.5), dim=-1)
        loss_rel = F.mse_loss(s_rel_code, packet.relation_code)
        
        return loss_soft + self.alpha * loss_feat + self.beta * loss_rel

    def step(self, loss, conf):
        # 根據信心度決定是否更新 Level 3 (Residual)
        self.optimizer.zero_grad()
        loss.backward()
        if conf > 0.5:
            self.optimizer.step()
