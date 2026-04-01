import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from misc.utils import is_using_distributed
from text_utils.tokenizer import tokenize
from .visual_transformer import visual_transformer
from .text_transformer import text_transformers
from .shared_modules import AllGather
from .CrossEmbeddingLayer_tse import VisualEmbeddingLayer, TexualEmbeddingLayer


class CLIP(nn.Module):
    def __init__(self, config, image_encode, text_encode, num_classes=11003, eps=1e-2):
        super().__init__()
        self.visual = image_encode
        self.encode_text = text_encode
        self.embed_dim = config.model.embed_dim

        self.use_gather = config.model.use_gather
        self.logit_scale = nn.Parameter(torch.ones([]))
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        self.config = config
        self.eps = eps

        model_cfg = getattr(config, "model", None)
        loss_cfg = getattr(config, "loss", None)
        baseline_cfg = getattr(model_cfg, "baseline", None) if model_cfg is not None else None
        weights_cfg = getattr(loss_cfg, "weights", None) if loss_cfg is not None else None

        self.use_global_align = bool(getattr(baseline_cfg, "global_align", True)) if baseline_cfg is not None else True
        self.use_local_align = bool(getattr(baseline_cfg, "local_align", True)) if baseline_cfg is not None else True
        self.use_id_loss = bool(getattr(baseline_cfg, "id_loss", True)) if baseline_cfg is not None else True

        self.lambda_global = float(getattr(weights_cfg, "global_align", 1.0)) if weights_cfg is not None else 1.0
        self.lambda_local = float(getattr(weights_cfg, "local_align", 1.0)) if weights_cfg is not None else 1.0
        self.lambda_id = float(getattr(weights_cfg, "id", 0.5)) if weights_cfg is not None else 0.5

        # caption-guided decoupling config (safe defaults)
        self.cap_heads = int(getattr(model_cfg, "cap_heads", 8)) if model_cfg is not None else 8
        self.view_heads = int(getattr(model_cfg, "view_heads", 8)) if model_cfg is not None else 8
        self.alpha_inv = float(getattr(model_cfg, "alpha_inv", 1.0)) if model_cfg is not None else 1.0
        self.beta_sp = float(getattr(model_cfg, "beta_sp", 1.0)) if model_cfg is not None else 1.0

        self.lambda_view = float(getattr(weights_cfg, "view", 1.0)) if weights_cfg is not None else 1.0
        self.lambda_dec = float(getattr(weights_cfg, "dec", 1.0)) if weights_cfg is not None else 1.0

        self.use_cap_view_decoupling = bool(getattr(model_cfg, "use_cap_view_decoupling", True)) if model_cfg is not None else True
        self.use_view_loss = bool(getattr(model_cfg, "use_view_loss", True)) if model_cfg is not None else True
        self.use_dec_loss = bool(getattr(model_cfg, "use_dec_loss", True)) if model_cfg is not None else True


        # modules for caption-guided branch
        self.cap_cross_attn = nn.MultiheadAttention(self.embed_dim, num_heads=self.cap_heads, batch_first=True)
        self.view_cross_attn = nn.MultiheadAttention(self.embed_dim, num_heads=self.view_heads, batch_first=True)
        self.view_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.attn_pool_fc = nn.Linear(self.embed_dim, 1)
        self.inv_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.sp_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.view_classifier = nn.Linear(self.embed_dim, 2)
        nn.init.normal_(self.view_classifier.weight.data, std=0.001)
        nn.init.constant_(self.view_classifier.bias.data, val=0.0)

        self.visual_emb_layer = VisualEmbeddingLayer(ratio=0.3)
        self.textual_emb_layer = TexualEmbeddingLayer(ratio=0.3)

        if self.use_id_loss:
            self.classifier = nn.Linear(self.embed_dim, num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)
        else:
            self.classifier = None

    # attention pooling over token sequence [B, T, D] -> [B, D]
    def attention_pool(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, D], mask: [B, T] with 1 for valid, 0 for pad
        scores = self.attn_pool_fc(x).squeeze(-1)  # [B, T]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        weights = torch.softmax(scores, dim=1)
        weights = weights.clamp(min=1e-6)
        weights = weights / weights.sum(dim=1, keepdim=True)
        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        return pooled

    def extract_view_specific_feature(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: [B, N, D]
        # single-query view token summarizes global view-specific nuisance
        q = self.view_token.to(patch_tokens.dtype).expand(patch_tokens.size(0), -1, -1)  # [B,1,D]
        out, _ = self.view_cross_attn(query=q, key=patch_tokens, value=patch_tokens)
        return out.squeeze(1)

    def extract_caption_guided_feature(self, text_tokens_dense: torch.Tensor, patch_tokens: torch.Tensor):
        # text_tokens_dense: [B, M, D], patch_tokens: [B, N, D]
        # F_sem_v is a token-level semantic visual representation (not a single embedding)
        if text_tokens_dense.dim() != 3 or patch_tokens.dim() != 3:
            raise ValueError("text_tokens_dense and patch_tokens must be 3D tensors")
        F_sem_v_attn, _ = self.cap_cross_attn(query=patch_tokens, key=text_tokens_dense, value=text_tokens_dense)
        F_sem_v = patch_tokens + F_sem_v_attn
        f_inv = self.inv_proj(self.attention_pool(F_sem_v))
        f_inv = F.normalize(f_inv, dim=-1)
        return F_sem_v, f_inv

    def build_final_identity_feature(self, f_cls: torch.Tensor, f_inv: torch.Tensor, f_sp: torch.Tensor):
        # f_cls: base global identity feature
        # f_inv: caption-guided invariant feature
        # f_sp: view-specific nuisance feature
        f_id_raw = f_cls + self.alpha_inv * f_inv - self.beta_sp * f_sp
        # normalize only final retrieval feature
        f_id = F.normalize(f_id_raw, dim=-1)
        return f_id_raw, f_id

    def _ensure_token_sequence(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if x is None or not torch.is_tensor(x) or x.dim() != 3:
            shape = None if x is None else tuple(x.shape)
            raise ValueError(f"{name} must be a 3D tensor [B, T, D], got: {shape}")
        return x

    def _split_visual_tokens(self, visual_tokens: torch.Tensor):
        # visual_tokens: [B, T, D]
        if visual_tokens.size(1) >= 2:
            visual_cls_token = visual_tokens[:, 0, :]
            visual_patch_tokens = visual_tokens[:, 1:, :]
        else:
            # fallback: treat all tokens as patches and use mean as global token
            visual_cls_token = visual_tokens.mean(dim=1)
            visual_patch_tokens = visual_tokens
        return visual_cls_token, visual_patch_tokens

    def _split_text_tokens(self, text_tokens_dense: torch.Tensor, text_global: torch.Tensor):
        # text_tokens_dense: [B, M, D], text_global: [B, D]
        caption_tokens_for_cross_attn = text_tokens_dense
        caption_global_feature = text_global
        return caption_tokens_for_cross_attn, caption_global_feature

    # decoupling loss between invariant and view-specific features
    def compute_decoupling_loss(self, f_inv, f_sp):
        f_inv_n = F.normalize(f_inv, dim=-1)
        f_sp_n = F.normalize(f_sp, dim=-1)
        cos_sim = F.cosine_similarity(f_inv_n, f_sp_n, dim=-1).abs()
        dec_loss = (cos_sim - 0.001).clamp(min=0.0).mean() * 100.0
        return dec_loss, dec_loss

    def forward(self, input, alpha, training=False):
        ret = dict()

        images = input['image'].to(self.config.device)
        texts = input.get('caption', None)
        ids = input.get('id', None)
        ids = ids.to(self.config.device) if ids is not None else None

        text_tokens = None
        if texts is not None:
            text_tokens = tokenize(texts, context_length=self.config.experiment.text_length).to(self.config.device)

        image_features, image_seq_embeddings, feaAttn_i = self._encode_image_backbone(
            images, return_dense=True, training=training
        )
        text_features, text_seq_embeddings, feaAttn_t = (None, None, None)
        if text_tokens is not None:
            text_features, text_seq_embeddings, feaAttn_t = self.encode_text(
                text_tokens, return_dense=True, training=training
            )

        # robust token handling
        visual_tokens = None
        visual_cls_from_tokens = None
        patch_tokens = None
        if image_seq_embeddings is not None:
            visual_tokens = self._ensure_token_sequence(image_seq_embeddings, "image_seq_embeddings")
            visual_cls_from_tokens, patch_tokens = self._split_visual_tokens(visual_tokens)

        f_cls = image_features if image_features is not None else visual_cls_from_tokens

        # prepare dense tokens
        # F_sem_v: token-level semantic visual representation [B, M, D]
        # f_inv: pooled caption-guided invariant feature [B, D]
        # f_sp: pooled view-specific feature [B, D]
        F_sem_v = None
        f_inv = None
        f_sp = None
        f_sp_proj = None
        f_id_raw = f_cls
        f_id = F.normalize(f_cls, dim=-1)

        caption_tokens_dense = None
        caption_global_feature = None
        if text_seq_embeddings is not None and text_features is not None:
            text_tokens_dense = self._ensure_token_sequence(text_seq_embeddings, "text_seq_embeddings")
            caption_tokens_dense, caption_global_feature = self._split_text_tokens(text_tokens_dense, text_features)

        if self.use_cap_view_decoupling and caption_tokens_dense is not None and patch_tokens is not None:
            F_sem_v, f_inv = self.extract_caption_guided_feature(caption_tokens_dense, patch_tokens)
            f_sp = self.extract_view_specific_feature(patch_tokens)
            f_sp_proj = F.normalize(self.sp_proj(f_sp), dim=-1)
            f_id_raw, f_id = self.build_final_identity_feature(f_cls, f_inv, f_sp_proj)

        ga_loss = torch.zeros((), device=f_cls.device)
        la_loss = torch.zeros((), device=f_cls.device)
        ln_itc = torch.zeros((), device=f_cls.device)
        lr_itc = torch.zeros((), device=f_cls.device)
        ln_itc_tse = torch.zeros((), device=f_cls.device)
        lr_itc_tse = torch.zeros((), device=f_cls.device)
        sim_targets = None
        logit_scale = None

        if text_features is not None and ids is not None:
            idx = ids.view(-1, 1)
            gathered_ids = self.all_gather(ids)
            idx_all = gathered_ids.view(1, -1)
            pos_idx = torch.eq(idx, idx_all).float()
            denom = pos_idx.sum(1, keepdim=True).clamp_min(1.0)
            sim_targets = pos_idx / denom

            logit_scale = self.logit_scale.exp().clamp(max=100)

        if self.use_global_align and text_features is not None and sim_targets is not None and logit_scale is not None:
            # use decoupled identity feature when available
            if f_id is not None:
                image_features_norm = F.normalize(f_id, dim=-1)
            else:
                image_features_norm = F.normalize(f_cls, dim=-1)
            text_features_norm = F.normalize(text_features, dim=-1)
            image_features_norm_gathered = self.all_gather(image_features_norm)
            text_features_norm_gathered = self.all_gather(text_features_norm)

            ga_loss, ln_itc, lr_itc = self.calc_contrastive(
                image_features_norm, text_features_norm,
                image_features_norm_gathered, text_features_norm_gathered,
                sim_targets, logit_scale
            )
            ret['ga_loss'] = ga_loss * self.config.experiment.nitc_ratio
            ret['ln_itc'] = ln_itc
            ret['lr_itc'] = lr_itc

        i_tse_f = None
        t_tse_f = None
        # local alignment still uses original dense visual tokens
        if self.use_local_align and feaAttn_i is not None and feaAttn_t is not None and sim_targets is not None and logit_scale is not None:
            i_tse_f = F.normalize(self.visual_emb_layer(feaAttn_i, image_seq_embeddings), dim=-1)
            t_tse_f = F.normalize(self.textual_emb_layer(feaAttn_t, text_tokens, text_seq_embeddings), dim=-1)
            i_tse_f_norm_gathered = self.all_gather(i_tse_f)
            t_tse_f_norm_gathered = self.all_gather(t_tse_f)
            i_tse_f_s_gathered = i_tse_f_norm_gathered.detach()
            t_tse_f_s_gathered = t_tse_f_norm_gathered.detach()
            la_loss, ln_itc_tse, lr_itc_tse = self.calc_contrastive(
                i_tse_f, t_tse_f,
                i_tse_f_norm_gathered, t_tse_f_norm_gathered,
                sim_targets, logit_scale
            )
            ret['la_loss'] = la_loss * self.config.experiment.nitc_ratio
            ret['ln_itc_tse'] = ln_itc_tse
            ret['lr_itc_tse'] = lr_itc_tse

        id_loss = torch.zeros((), device=f_cls.device)
        image_logits = None
        text_logits = None
        if self.use_id_loss and self.classifier is not None and text_features is not None and ids is not None:
            image_logits = self.classifier(f_id_raw if f_id_raw is not None else f_cls)
            text_logits = self.classifier(text_features)
            id_loss = (F.cross_entropy(image_logits, ids) + F.cross_entropy(text_logits, ids)) / 2
            ret['id_loss'] = id_loss

        if image_logits is not None:
            ret['id_logits'] = image_logits
        if text_logits is not None:
            ret['text_logits'] = text_logits

        total = torch.zeros((), device=f_cls.device)
        if self.use_global_align:
            total = total + self.lambda_global * ga_loss
        if self.use_local_align:
            total = total + self.lambda_local * la_loss
        if self.use_id_loss:
            total = total + self.lambda_id * id_loss

        # new losses
        view_loss = torch.zeros((), device=f_cls.device)
        dec_loss = torch.zeros((), device=f_cls.device)
        dec_inv_sp = torch.zeros((), device=f_cls.device)

        view_labels = input.get('cam_id', None)
        if self.use_cap_view_decoupling and self.use_view_loss and view_labels is not None and f_sp_proj is not None:
            # expected binary labels: 0=aerial, 1=ground
            view_labels = view_labels.to(f_cls.device).long()
            view_logits = self.view_classifier(f_sp_proj)
            view_loss = F.cross_entropy(view_logits, view_labels)
            view_pred = view_logits.argmax(dim=1)
            view_acc = (view_pred == view_labels).float().mean()
            ret['view_acc'] = view_acc
            ret['view_logits'] = view_logits

        if self.use_cap_view_decoupling and self.use_dec_loss and f_inv is not None and f_sp_proj is not None:
            dec_inv_sp, dec_loss = self.compute_decoupling_loss(f_inv, f_sp_proj)

        total = total + self.lambda_view * view_loss + self.lambda_dec * dec_loss

        ret['total_loss'] = total
        ret['f_cls'] = f_cls
        if text_features is not None:
            ret['t_eos'] = text_features
        if i_tse_f is not None:
            ret['local_img_feat'] = i_tse_f
        if t_tse_f is not None:
            ret['local_txt_feat'] = t_tse_f

        if f_id_raw is not None:
            ret['f_id_raw'] = f_id_raw
        if f_id is not None:
            ret['f_id'] = f_id
        if f_inv is not None:
            ret['f_inv'] = f_inv
        if f_sp is not None:
            ret['f_sp'] = f_sp
        if f_sp_proj is not None:
            ret['f_sp_proj'] = f_sp_proj
        if F_sem_v is not None:
            ret['F_sem_v'] = F_sem_v
        ret['view_loss'] = view_loss
        ret['dec_inv_sp'] = dec_inv_sp
        ret['dec_loss'] = dec_loss

        return ret

    def calc_contrastive(self, image_features, text_features,
                         image_features_gathered, text_features_gathered,
                         sim_targets, logit_scale):
        sim_i2t = logit_scale * image_features @ text_features_gathered.t()
        sim_t2i = logit_scale * text_features @ image_features_gathered.t()
        bsz = sim_i2t.size(0)

        def reverse_ce(logits: torch.Tensor):
            b = logits.size(0)
            if b <= 1:
                return logits.new_tensor(0.0)
            eye = torch.eye(b, device=logits.device)
            neg_mask = 1.0 - eye
            denom = neg_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            neg_target = neg_mask / denom
            log_prob = F.log_softmax(logits, dim=1)
            loss = -(neg_target * log_prob).sum(dim=1).mean()
            return loss

        if sim_i2t.size(1) == bsz:
            labels = torch.arange(bsz, device=sim_i2t.device)
            loss_i2t = F.cross_entropy(sim_i2t, labels)
            loss_t2i = F.cross_entropy(sim_t2i, labels)
            ln_itc = 0.5 * (loss_i2t + loss_t2i)
        else:
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()
            ln_itc = 0.5 * (loss_i2t + loss_t2i)

        lr_i2t = reverse_ce(sim_i2t)
        lr_t2i = reverse_ce(sim_t2i)
        lr_itc = 0.5 * (lr_i2t + lr_t2i)

        itc_loss = ln_itc + lr_itc
        return itc_loss, ln_itc, lr_itc

    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except Exception:
            return self.encode_text.text_projection.weight.dtype

    def encode_image(self, image, cam_id=None, training=False, return_dense=False):
        if return_dense:
            return self.visual(image.type(self.dtype), return_dense=return_dense, training=training)
        output = self.visual(image.type(self.dtype), training=training)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output

    def _encode_image_backbone(self, image, cam_id=None, training=False, return_dense=False):
        if return_dense:
            output = self.visual(image.type(self.dtype), cam_id=cam_id, training=training, return_dense=return_dense)
            if isinstance(output, (tuple, list)):
                f_cls, dense_feat, attn = output
            else:
                f_cls, dense_feat, attn = output, None, None
            return f_cls, dense_feat, attn
        output = self.visual(image.type(self.dtype), training=training)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output, None, None

    def all_gather(self, input):
        if not self.use_gather or not is_using_distributed():
            return input
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output


def clip_vitb(config, num_classes=11003):
    image_encode = visual_transformer(config)
    text_encode = text_transformers(config)
    model = CLIP(config, image_encode, text_encode, num_classes, config.experiment.ritc_eps)
    return model
