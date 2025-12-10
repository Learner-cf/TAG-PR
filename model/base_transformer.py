from collections import OrderedDict

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
import torch.nn.functional as F
from einops import rearrange, reduce
global LAYER_NORM
LAYER_NORM = True


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        if LAYER_NORM:
            ret = super().forward(x)
        else:
            ret = x
        return ret


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model * 4)
        self.gelu = QuickGELU()
        self.c_proj = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout: float = 0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = MLP(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, need_weights: bool = False):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=need_weights, attn_mask=self.attn_mask)

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        attn_out, attn_weight = self.attention(self.ln_1(x), need_weights=need_weights)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        if need_weights:
            return x, attn_weight
        return x


class ResidualAttentionBlockMoE(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout: float = 0.,
                 num_of_experts: int = None, num_of_selected_experts: int = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        self.router = nn.Linear(d_model, 2)

        self.num_of_experts = num_of_experts
        self.num_of_selected_experts = num_of_selected_experts
        self.img_router = nn.Linear(d_model, 2)
        nn.init.normal_(self.img_router.weight.data, std=0.001)
        nn.init.constant_(self.img_router.bias.data, val=0.0)

        self.fea_router = nn.Linear(d_model, num_of_experts)
        nn.init.normal_(self.fea_router.weight.data, std=0.001)
        nn.init.constant_(self.fea_router.bias.data, val=0.0)

        self.experts = nn.ModuleList([
            MLP(d_model) for _ in range(self.num_of_experts)
        ])
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, need_weights: bool = False):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=need_weights, attn_mask=self.attn_mask)

    def forward_experts(self, x: torch.Tensor, cam_id=None, training=False):
        # reshape [N, B, D] -> [B, N, D]
        x = x.permute(1, 0, 2)  # [B, N, D]
        B, N, D = x.shape
        x_flat = x.reshape(-1, D)  # [B*N, D]
    
        view_loss, view_acc, ortho_loss = torch.zeros(1), torch.zeros(1), torch.zeros(1)
        view_logits = self.img_router(x[:, -1, :])  # [B, E]
    
        if training and cam_id is not None:
            ce = nn.CrossEntropyLoss()
            view_loss = ce(view_logits, cam_id)
            preds = torch.argmax(view_logits, dim=1)
            correct = (preds == cam_id).sum().item()
            total = cam_id.size(0)
            view_acc = correct / total
            cos_sim = torch.cosine_similarity(F.normalize(x[:, 0, :], dim=-1), F.normalize(x[:, -1, :]), dim=-1).abs()
            ortho_loss = (cos_sim - 0.001).clamp(min=0.0).mean() * 100
    
        group_probs = F.gumbel_softmax(view_logits, hard=not training)
        group_id = torch.argmax(group_probs, dim=1)  # [B]
    
        # logits
        gate_logits = self.fea_router(x_flat)  # [B*N, E]
    
        # construct expert_mask
        expert_mask = torch.ones((B, N, self.num_of_experts), device=x.device) * -1e9
        for b in range(B):
            if group_id[b] == 0:
                expert_mask[b, :, :self.num_of_experts - 1] = 0
            else:
                expert_mask[b, :, 1:] = 0
        expert_mask_flat = expert_mask.reshape(-1, self.num_of_experts)  # [B*N, E]
    
        # Add noise to gate logits during training
        if training:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * 0.01
    
        gate_softmax = F.softmax(gate_logits + expert_mask_flat, dim=-1)  # [B*N, E]
        weights, selected = torch.topk(gate_softmax, k=self.num_of_selected_experts, dim=-1)  # [B*N, K]
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)  # normalize
    
        # construct one-hot mask
        one_hot = F.one_hot(selected, num_classes=self.num_of_experts).float()  # [B*N, K, E]
    
        # Expand weights and combine with one-hot
        weights_expanded = weights.unsqueeze(-1)  # [B*N, K, 1]
        combined_weights = (weights_expanded * one_hot).sum(dim=1)  # [B*N, E]
    
        # get expert outputs
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=-1)  # [B*N, D, E]
    
        # weight and sum outputs
        final_output_flat = torch.einsum('be,bde->bd', combined_weights, expert_outputs)  # [B*N, D]
    
        # Reshape final output back to original shape
        final_output = final_output_flat.reshape(B, N, D).permute(1, 0, 2)  # [N, B, D]
    
        return final_output, view_loss, view_acc, ortho_loss

    def forward(self, x: torch.Tensor, cam_id=None, need_weights=False, training=False):
        attn_out, attn_weight = self.attention(self.ln_1(x), need_weights=need_weights)
        x = x + attn_out
        res, view_loss, view_acc, ortho_loss = self.forward_experts(self.ln_2(x), cam_id, training=training)
        x = x + res
        if need_weights:
            return x, view_loss, view_acc, ortho_loss, attn_weight
        return x, view_loss, view_acc, ortho_loss


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, checkpoint: bool = False,
                 dropout: float = 0., emb_dropout: float = 0., expert_blocks: list = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.checkpoint = checkpoint
        self.dropout = nn.Dropout(emb_dropout)
        # self.resblocks = nn.Sequential(
        #     *[ResidualAttentionBlock(width, heads, attn_mask, dropout=dropout) for _ in range(layers)])
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads, attn_mask, dropout=dropout) for _ in range(layers)])
        if expert_blocks:
            self.expert_blocks = expert_blocks

    def checkpoint_fwd(self, layer, input, segments=2):
        """checkpoint forward"""
        # Make sure that the input to checkpoint have requires_grad=True, so that
        # the autograd can take care of the checkpointed part of model
        if not input.requires_grad:
            input = input.detach()
            input.requires_grad = True
        return checkpoint_sequential(layer, segments, input)

    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        if self.checkpoint:
            return self.checkpoint_fwd(self.resblocks, x, self.layers)

        for i, blk in enumerate(self.resblocks):
            if i == len(self.resblocks) - 1:
                x, attn_weight = blk(x, need_weights=True)
            else:
                x = blk(x)

        return x, attn_weight


class Transformer_MoE(nn.Module):
    def __init__(self, config, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, checkpoint: bool = False,
                 dropout: float = 0., emb_dropout: float = 0.):
        super().__init__()
        self.width = width
        self.layers = layers
        self.checkpoint = checkpoint
        self.dropout = nn.Dropout(emb_dropout)

        self.expert_blocks = getattr(config, 'expert_blocks', [])
        self.num_of_experts = getattr(config, 'num_of_experts', 0)
        self.num_of_selected_experts = getattr(config, 'num_of_selected_experts', 0)

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlockMoE(width, heads, attn_mask, dropout=dropout, num_of_experts=self.num_of_experts,
                                      num_of_selected_experts=self.num_of_selected_experts)
            if i in self.expert_blocks else
            ResidualAttentionBlock(width, heads, attn_mask, dropout=dropout)
            for i in range(layers)
        ])
        # self.resblocks = nn.Sequential(
        #     *[ResidualAttentionBlock(width, heads, attn_mask, dropout=dropout) for _ in range(layers)])

    def checkpoint_fwd(self, layer, input, segments=2):
        """checkpoint forward"""
        # Make sure that the input to checkpoint have requires_grad=True, so that
        # the autograd can take care of the checkpointed part of model
        if not input.requires_grad:
            input = input.detach()
            input.requires_grad = True
        return checkpoint_sequential(layer, segments, input)

    def forward(self, x: torch.Tensor, cam_id=None, training=False):
        losses = {}
        view_losses = []
        view_accs = []
        ortho_losses = []
        x = self.dropout(x)
        if self.checkpoint:
            return self.checkpoint_fwd(self.resblocks, x, self.layers)
        for i, blk in enumerate(self.resblocks):
            # MoE
            if self.expert_blocks and i in self.expert_blocks:
                # last block attention weights for tse
                if i == len(self.resblocks) - 1:
                    x, view_loss, view_acc, ortho_loss, attn_weight = blk(x, cam_id=cam_id, training=training,
                                                                          need_weights=True)
                else:
                    x, view_loss, view_acc, ortho_loss = blk(x, cam_id=cam_id, training=training)
                view_losses.append(view_loss)
                view_accs.append(view_acc)
                ortho_losses.append(ortho_loss)
                # remove the view token
                if i == self.expert_blocks[-1]:
                    x = x[:-1, :, :]
            # non-MoE
            else:
                if i == len(self.resblocks) - 1:
                    x, attn_weight = blk(x, need_weights=True)
                else:
                    x = blk(x)
        if training:
            losses['view_loss'] = torch.stack(view_losses).mean()
            losses['view_acc'] = sum(view_accs)/len(view_accs)
            losses['ortho_loss'] = torch.stack(ortho_losses).mean()
        return x, losses, attn_weight

