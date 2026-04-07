import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class DistillPacket:
    """Universal Knowledge Port 封裝協議"""
    soft_logits: torch.Tensor
    feat_summary: torch.Tensor
    relation_code: torch.Tensor
    attn_summary: Optional[torch.Tensor] = None
    confidence_score: float = 1.0


class UDLSLinear(nn.Module):
    """
    基於 W ≈ Decode(C, S, R, A) 的權重表示層
    C: code indices
    S: group scales
    R: dense residual
    codebook: anchor vectors
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        codebook_size: int = 256,
        group_size: int = 8,
        bias: bool = True,
        residual_init_scale: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.codebook_size = codebook_size

        total_elems = in_features * out_features
        if total_elems % group_size != 0:
            raise ValueError(
                f"in_features*out_features={total_elems} 必須可被 group_size={group_size} 整除"
            )

        self.num_groups = total_elems // group_size

        self.codebook = nn.Parameter(
            torch.randn(codebook_size, group_size) * 0.02
        )

        self.register_buffer(
            "C",
            torch.randint(0, codebook_size, (self.num_groups,), dtype=torch.long)
        )

        self.S = nn.Parameter(torch.ones(self.num_groups))

        self.R = nn.Parameter(
            torch.randn(out_features, in_features) * residual_init_scale
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def decode_weight(self) -> torch.Tensor:
        w_base = self.codebook[self.C] * self.S.unsqueeze(-1)
        w_base = w_base.reshape(self.out_features, self.in_features)
        return w_base + self.R

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.decode_weight()
        return F.linear(x, weight, self.bias)

    @torch.no_grad()
    def refresh_codes(self):
        """
        可選：根據目前 (W - R) 重新指派最近 codebook index。
        這不是必要步驟，但若你想讓 C 真正跟著學到的 codebook 改善，可週期性呼叫。
        """
        target = (-self.R).reshape(self.num_groups, self.group_size)
        # 更合理的是根據某個目標 full weight 做 assignment；這裡僅提供原型接口。
        dist = torch.cdist(target, self.codebook)
        self.C.copy_(dist.argmin(dim=1))


class UniversalPort(nn.Module):
    """跨架構蒸餾接口：將 Teacher 訊號標準化"""
    def __init__(self, teacher_dim: int, student_dim: int):
        super().__init__()
        self.projector = nn.Linear(teacher_dim, student_dim)

    def forward(
        self,
        t_logits: torch.Tensor,
        t_hidden: torch.Tensor,
        t_attn: Optional[torch.Tensor] = None,
        conf: float = 1.0
    ) -> DistillPacket:
        feat_summary = self.projector(t_hidden)

        rel = torch.bmm(t_hidden, t_hidden.transpose(1, 2))
        rel_code = F.softmax(rel / (t_hidden.size(-1) ** 0.5), dim=-1)

        attn_summary = None
        if t_attn is not None:
            attn_summary = t_attn

        return DistillPacket(
            soft_logits=t_logits,
            feat_summary=feat_summary,
            relation_code=rel_code,
            attn_summary=attn_summary,
            confidence_score=float(conf)
        )


class KDStudentWrapper(nn.Module):
    """
    包裝 student model，統一回傳 logits 與 hidden features。
    你可把自己的 backbone 放進來。
    """
    def __init__(self, backbone: nn.Module, hidden_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        預期 backbone 回傳:
        - 若是 tuple: (hidden, logits)
        - 若是單 tensor: 視為 hidden，再走一個 head
        你可以依自己模型改寫這段。
        """
        out = self.backbone(x)

        if isinstance(out, tuple) and len(out) == 2:
            hidden, logits = out
        else:
            raise ValueError("backbone 必須回傳 (hidden, logits)")

        return logits, hidden


class UDLSTrainer:
    """
    五維 Loss 動態平衡優化器
    - soft target KL
    - feature align
    - relation align
    - optional attention align
    - hard label CE
    """
    def __init__(
        self,
        model: nn.Module,
        temperature: float = 2.0,
        soft_weight: float = 0.25,
        ce_weight: float = 0.75,
        feat_weight: float = 1.0,
        rel_weight: float = 0.5,
        attn_weight: float = 0.0,
        lr_codebook: float = 1e-3,
        lr_residual: float = 1e-4,
        weight_decay: float = 1e-2,
        residual_update_threshold: float = 0.5
    ):
        self.model = model
        self.temperature = temperature
        self.soft_weight = soft_weight
        self.ce_weight = ce_weight
        self.feat_weight = feat_weight
        self.rel_weight = rel_weight
        self.attn_weight = attn_weight
        self.residual_update_threshold = residual_update_threshold

        codebook_params, residual_params, other_params = self._collect_params(model)

        param_groups = []
        if len(codebook_params) > 0:
            param_groups.append({
                "params": codebook_params,
                "lr": lr_codebook,
                "weight_decay": weight_decay
            })
        if len(residual_params) > 0:
            param_groups.append({
                "params": residual_params,
                "lr": lr_residual,
                "weight_decay": weight_decay
            })
        if len(other_params) > 0:
            param_groups.append({
                "params": other_params,
                "lr": lr_codebook,
                "weight_decay": weight_decay
            })

        self.optimizer = torch.optim.AdamW(param_groups)
        self.ce_loss_fn = nn.CrossEntropyLoss()

        self.codebook_params = codebook_params
        self.residual_params = residual_params

    def _collect_params(self, model: nn.Module):
        codebook_params: List[nn.Parameter] = []
        residual_params: List[nn.Parameter] = []
        codebook_param_ids = set()
        residual_param_ids = set()

        for module in model.modules():
            if isinstance(module, UDLSLinear):
                for p in [module.codebook, module.S]:
                    codebook_params.append(p)
                    codebook_param_ids.add(id(p))
                residual_params.append(module.R)
                residual_param_ids.add(id(module.R))
                if module.bias is not None:
                    codebook_params.append(module.bias)
                    codebook_param_ids.add(id(module.bias))

        other_params = []
        for p in model.parameters():
            if id(p) not in codebook_param_ids and id(p) not in residual_param_ids:
                other_params.append(p)

        return codebook_params, residual_params, other_params

    def compute_loss(
        self,
        s_logits: torch.Tensor,
        s_feat: torch.Tensor,
        labels: torch.Tensor,
        packet: DistillPacket,
        s_attn: Optional[torch.Tensor] = None
    ):
        T = self.temperature

        loss_soft = F.kl_div(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(packet.soft_logits / T, dim=-1),
            reduction="batchmean"
        ) * (T * T)

        loss_ce = self.ce_loss_fn(s_logits, labels)

        if s_feat.shape != packet.feat_summary.shape:
            raise ValueError(
                f"student feature shape {s_feat.shape} != teacher projected shape {packet.feat_summary.shape}"
            )
        loss_feat = F.mse_loss(s_feat, packet.feat_summary)

        s_rel = torch.bmm(s_feat, s_feat.transpose(1, 2))
        s_rel_code = F.softmax(s_rel / (s_feat.size(-1) ** 0.5), dim=-1)
        loss_rel = F.mse_loss(s_rel_code, packet.relation_code)

        loss_attn = torch.tensor(0.0, device=s_logits.device)
        if self.attn_weight > 0 and s_attn is not None and packet.attn_summary is not None:
            if s_attn.shape != packet.attn_summary.shape:
                raise ValueError(
                    f"student attn shape {s_attn.shape} != teacher attn shape {packet.attn_summary.shape}"
                )
            loss_attn = F.mse_loss(s_attn, packet.attn_summary)

        total_loss = (
            self.soft_weight * loss_soft +
            self.ce_weight * loss_ce +
            self.feat_weight * loss_feat +
            self.rel_weight * loss_rel +
            self.attn_weight * loss_attn
        )

        loss_dict = {
            "total": total_loss,
            "soft": loss_soft.detach(),
            "ce": loss_ce.detach(),
            "feat": loss_feat.detach(),
            "rel": loss_rel.detach(),
            "attn": loss_attn.detach()
        }
        return total_loss, loss_dict

    def _set_requires_grad(self, params: List[nn.Parameter], flag: bool):
        for p in params:
            p.requires_grad_(flag)

    def step(self, loss: torch.Tensor, conf: float):
        self.optimizer.zero_grad(set_to_none=True)

        if conf > self.residual_update_threshold:
            self._set_requires_grad(self.residual_params, True)
        else:
            self._set_requires_grad(self.residual_params, False)

        loss.backward()
        self.optimizer.step()

        if conf <= self.residual_update_threshold:
            self._set_requires_grad(self.residual_params, True)

    @torch.no_grad()
    def refresh_all_codes(self):
        for module in self.model.modules():
            if isinstance(module, UDLSLinear):
                module.refresh_codes()
