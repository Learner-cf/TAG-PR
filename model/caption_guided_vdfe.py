import torch
from torch import nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + FFN with residuals."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        self.ln_out = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        # query: [B, Nq, D], key_value: [B, Nk, D]
        q = self.ln_q(query)
        kv = self.ln_kv(key_value)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = query + attn_out
        x = x + self.ffn(self.ln_out(x))
        return x


class CaptionGuidedAGViewDecoupler(nn.Module):
    """
    Caption-guided aerial-ground view decoupler.
    Inputs:
      f_img: [B, D]
      text_seq: [B, L, D]
    Outputs:
      f_id: [B, D]
      f_view: [B, D]
      f_hat: [B, D]
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        lambda_rm: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.lambda_rm = float(lambda_rm)
        self.id_attn = CrossAttentionBlock(embed_dim, num_heads, dropout=dropout)
        self.view_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ln_id = nn.LayerNorm(embed_dim)

    def forward(self, f_img: torch.Tensor, text_seq: torch.Tensor):
        if f_img.dim() != 2:
            raise ValueError("f_img must be [B, D]")
        if text_seq.dim() != 3:
            raise ValueError("text_seq must be [B, L, D]")

        v = f_img.unsqueeze(1)  # [B,1,D]
        f_id = self.id_attn(v, text_seq)
        f_id = self.ln_id(f_id).squeeze(1)  # [B,D]

        r = f_img - f_id
        f_view = self.view_proj(r)  # [B,D]
        # normalize with epsilon to avoid NaN when f_view norm is tiny
        u = f_view / (f_view.norm(dim=-1, keepdim=True) + 1e-6)
        alpha = torch.sum(f_img * u, dim=-1, keepdim=True)  # [B,1]
        f_hat = f_img - self.lambda_rm * alpha * u
        return f_id, f_view, f_hat
