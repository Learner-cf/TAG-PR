import random
import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import copy

from misc import utils
from misc.utils import is_using_distributed
from text_utils.tokenizer import tokenize
from .visual_transformer import visual_transformer
from .text_transformer import text_transformers
from .base_transformer import Transformer, LayerNorm, QuickGELU

from .shared_modules import AllGather
from collections import OrderedDict

from .CrossEmbeddingLayer_tse import VisualEmbeddingLayer, TexualEmbeddingLayer


class ResidualViewDecomposer(nn.Module):
    """Text-anchored residual view decoupling."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, f_img: torch.Tensor, f_text: torch.Tensor):
        # normalize anchors to stabilize residual scale
        f_img_norm = F.normalize(f_img, dim=-1)
        f_text_norm = F.normalize(f_text, dim=-1)
        residual_input = f_img_norm - self.alpha * f_text_norm
        v = self.proj(residual_input)
        v = F.normalize(v, dim=-1)
        m_text = F.normalize(f_img - v, dim=-1)
        return m_text, v


class SharedImageHead(nn.Module):
    """Image-only shared head for inference consistency."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, f_img: torch.Tensor):
        return self.proj(f_img)


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

        self.visual_emb_layer = VisualEmbeddingLayer(ratio=0.3)
        self.textual_emb_layer = TexualEmbeddingLayer(ratio=0.3)

        self.use_trvd = bool(getattr(config.model, "use_trvd", False))
        self.use_residual_decomposer = bool(getattr(config.model, "use_residual_decomposer", True))
        self.use_shared_img_head = bool(getattr(config.model, "use_shared_img_head", True))
        self.use_view_classifier = bool(getattr(config.model, "use_view_classifier", True))
        if self.use_trvd:
            if self.use_residual_decomposer:
                self.trvd = ResidualViewDecomposer(self.embed_dim)
            if self.use_shared_img_head:
                self.shared_head_img = SharedImageHead(self.embed_dim)
            if self.use_view_classifier:
                self.view_classifier = nn.Linear(self.embed_dim, 2)
                nn.init.normal_(self.view_classifier.weight.data, std=0.001)
                nn.init.constant_(self.view_classifier.bias.data, val=0.0)

        if config.experiment.id:
            self.classifier = nn.Linear(self.embed_dim, num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

    def forward(self, input, alpha, training=False):
        ret = dict()

        images = input['image'].to(self.config.device)
        texts = input['caption']
        cam_id = input['cam_id'].to(self.config.device)

        text_tokens = tokenize(texts, context_length=self.config.experiment.text_length).to(self.config.device)
        ids = input['id'].to(self.config.device)

        image_features, image_seq_embeddings, feaAttn_i = self.encode_image(
            images, cam_id=cam_id, return_dense=True, training=training
        )
        text_features, text_seq_embeddings, feaAttn_t = self.encode_text(
            text_tokens, return_dense=True, training=training
        )
        image_features_norm = F.normalize(image_features)
        text_features_norm = F.normalize(text_features)
        image_features_norm_gathered = self.all_gather(image_features_norm)
        text_features_norm_gathered = self.all_gather(text_features_norm)
        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)

        idx = ids.view(-1, 1)
        gathered_ids = self.all_gather(ids)
        idx_all = gathered_ids.view(1, -1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        image_features_s_norm = image_features_norm.detach()
        text_features_s_norm = text_features_norm.detach()
        image_features_s_norm_gathered = image_features_norm_gathered.detach()
        text_features_s_norm_gathered = text_features_norm_gathered.detach()

        # ---- TRVD branch: compute m_text, v, m_img ----
        if self.use_trvd:
            if self.use_residual_decomposer:
                m_text, v = self.trvd(image_features, text_features)
            else:
                m_text = F.normalize(image_features, dim=-1)
                v = None

            if self.use_shared_img_head:
                m_img = self.shared_head_img(F.normalize(image_features, dim=-1))
                m_img = F.normalize(m_img, dim=-1)
            else:
                m_img = F.normalize(image_features, dim=-1)
        else:
            m_text, v, m_img = None, None, None

        # ---- Global alignment ----
        if self.use_trvd:
            loss_cfg = getattr(self.config, "loss", None)
            use_ga_on_m_text = getattr(loss_cfg, "use_ga_on_m_text", True) if loss_cfg is not None else True
            ga_input = m_text if use_ga_on_m_text else m_img
            ga_input_g = self.all_gather(ga_input)
            ga_input_s = ga_input.detach()
            ga_input_s_g = ga_input_g.detach()
            ga_loss, ln_itc, lr_itc = self.calc_contrastive(ga_input, text_features_norm, ga_input_s,
                                                            text_features_s_norm, ga_input_g,
                                                            text_features_norm_gathered, ga_input_s_g,
                                                            text_features_s_norm_gathered, sim_targets, alpha, logit_scale)
        else:
            ga_loss, ln_itc, lr_itc = self.calc_contrastive(image_features_norm, text_features_norm, image_features_s_norm,
                                                            text_features_s_norm, image_features_norm_gathered,
                                                            text_features_norm_gathered, image_features_s_norm_gathered,
                                                            text_features_s_norm_gathered, sim_targets, alpha, logit_scale)

        i_tse_f = F.normalize(self.visual_emb_layer(feaAttn_i, image_seq_embeddings), dim=-1)
        t_tse_f = F.normalize(self.textual_emb_layer(feaAttn_t, text_tokens, text_seq_embeddings), dim=-1)
        i_tse_f_norm_gathered = self.all_gather(i_tse_f)
        t_tse_f_norm_gathered = self.all_gather(t_tse_f)
        i_tse_f_s_gathered = i_tse_f_norm_gathered.detach()
        t_tse_f_s_gathered = t_tse_f_norm_gathered.detach()
        la_loss, ln_itc_tse, lr_itc_tse = self.calc_contrastive(i_tse_f, t_tse_f, i_tse_f.detach(), t_tse_f.detach(),
                                                               i_tse_f_norm_gathered, t_tse_f_norm_gathered,
                                                               i_tse_f_s_gathered, t_tse_f_s_gathered, sim_targets, alpha, logit_scale)

        ret['ga_loss'] = ga_loss * self.config.experiment.nitc_ratio
        ret['la_loss'] = la_loss * self.config.experiment.nitc_ratio
        ret['itc_tse_loss'] = ret['la_loss']
        ret['itc_loss'] = ret['ga_loss'] + ret['la_loss']
        ret['ln_itc'] = ln_itc
        ret['lr_itc'] = lr_itc
        ret['ln_itc_tse'] = ln_itc_tse
        ret['lr_itc_tse'] = lr_itc_tse


        if self.config.experiment.id:
            if self.use_trvd:
                loss_cfg = getattr(self.config, "loss", None)
                use_id_on_m_img = getattr(loss_cfg, "use_id_on_m_img", True) if loss_cfg is not None else True
                id_input = m_img if use_id_on_m_img else m_text
                image_logits = self.classifier(id_input)
            else:
                image_logits = self.classifier(image_features)
            text_logits = self.classifier(text_features)
            id_loss = (F.cross_entropy(image_logits, ids) + F.cross_entropy(text_logits, ids)) / 2
            ret['id_loss'] = id_loss * self.config.experiment.id_ratio

        if self.use_trvd:
            loss_cfg = getattr(self.config, "loss", None)
            use_view_loss = getattr(loss_cfg, "use_view_loss", True) if loss_cfg is not None else True
            use_orth_loss = getattr(loss_cfg, "use_orth_loss", True) if loss_cfg is not None else True
            use_cons_loss = getattr(loss_cfg, "use_cons_loss", True) if loss_cfg is not None else True
            use_bridge = getattr(loss_cfg, "use_img_text_consistency", True) if loss_cfg is not None else True

            # view loss
            if use_view_loss and self.use_view_classifier and v is not None:
                view_logits = self.view_classifier(v)
                view_loss = F.cross_entropy(view_logits, cam_id)
                preds = torch.argmax(view_logits, dim=1)
                view_acc = (preds == cam_id).float().mean().item()
                ret['view_loss'] = view_loss
                ret['view_acc'] = view_acc
            else:
                ret['view_loss'] = torch.zeros((), device=image_features.device)
                ret['view_acc'] = 0.0

            # orthogonality loss
            if use_orth_loss and v is not None:
                ret['orth_loss'] = F.cosine_similarity(m_text, v, dim=-1).abs().mean()
            else:
                ret['orth_loss'] = torch.zeros((), device=image_features.device)

            # shared consistency: same ID but different cam_id (default on m_img)
            if use_cons_loss:
                with torch.no_grad():
                    ids_col = ids.view(-1, 1)
                    cam_col = cam_id.view(-1, 1)
                    same_id = ids_col.eq(ids_col.t())
                    diff_cam = cam_col.ne(cam_col.t())
                    mask = same_id & diff_cam
                    mask = torch.triu(mask, diagonal=1)
                if mask.any():
                    dist = torch.norm(m_img.unsqueeze(1) - m_img.unsqueeze(0), dim=-1)
                    ret['cons_loss'] = dist[mask].mean()
                else:
                    ret['cons_loss'] = torch.zeros((), device=image_features.device)
            else:
                ret['cons_loss'] = torch.zeros((), device=image_features.device)

            # image-text consistency bridge
            if use_bridge:
                ret['bridge_loss'] = (1 - F.cosine_similarity(m_img, m_text, dim=-1)).mean()
            else:
                ret['bridge_loss'] = torch.zeros((), device=image_features.device)

            # debug metrics
            ret['alpha'] = self.trvd.alpha.detach().item() if self.use_residual_decomposer else 0.0
            ret['m_text_norm_mean'] = m_text.norm(dim=-1).mean().detach().item()
            ret['m_img_norm_mean'] = m_img.norm(dim=-1).mean().detach().item()
            ret['v_norm_mean'] = v.norm(dim=-1).mean().detach().item() if v is not None else 0.0

            # total loss aggregation (safe defaults)
            lambda_view = getattr(loss_cfg, "lambda_view", 1.0) if loss_cfg is not None else 1.0
            lambda_orth = getattr(loss_cfg, "lambda_orth", 0.1) if loss_cfg is not None else 0.1
            lambda_cons = getattr(loss_cfg, "lambda_cons", 0.5) if loss_cfg is not None else 0.5
            lambda_bridge = getattr(loss_cfg, "lambda_img_text_consistency", 1.0) if loss_cfg is not None else 1.0
            total = ret.get('ga_loss', 0) + ret.get('la_loss', 0) + ret.get('id_loss', 0)
            total = total + lambda_view * ret.get('view_loss', 0)
            total = total + lambda_orth * ret.get('orth_loss', 0)
            total = total + lambda_cons * ret.get('cons_loss', 0)
            total = total + lambda_bridge * ret.get('bridge_loss', 0)
            ret['total_loss'] = total
        elif 'total_loss' not in ret:
            ret['total_loss'] = ret.get('itc_loss', 0) + ret.get('id_loss', 0)

        return ret

    def compute_ritc(self, img_feats, txt_feats, img_feats_gathered, txt_feats_gathered, logit_scale, sim_targets):
        logits_per_img = logit_scale * img_feats @ txt_feats_gathered.t()
        logits_per_txt = logit_scale * txt_feats @ img_feats_gathered.t()
        img_log = F.log_softmax(logits_per_img, dim=1)
        txt_log = F.log_softmax(logits_per_txt, dim=1)
        target_log = (sim_targets + self.eps).log()
        kl_img = F.kl_div(target_log, img_log, log_target=True, reduction='batchmean')
        kl_txt = F.kl_div(target_log, txt_log, log_target=True, reduction='batchmean')
        ritc_loss = 0.5 * (kl_img + kl_txt)
        return ritc_loss

    def compute_sdm(self, image_features, text_features, image_features_gathered, text_features_gathered,
                    labels, logit_scale, epsilon=1e-8):
        """
        Similarity Distribution Matching
        """

        t2i_cosine_theta = text_features @ image_features_gathered.t()
        i2t_cosine_theta = image_features @ text_features_gathered.t()

        text_proj_image = logit_scale * t2i_cosine_theta
        image_proj_text = logit_scale * i2t_cosine_theta

        i2t_pred = F.softmax(image_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels + epsilon))
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels + epsilon))

        loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
        return loss

    # input features are normed
    def calc_contrastive(self, image_features, text_features, image_features_s, text_features_s,
                         image_features_gathered, text_features_gathered, image_features_s_gathered,
                         text_features_s_gathered,
                         sim_targets, alpha, logit_scale):
        # LN-ITC + LR-ITC (symmetric; no standalone RITC)
        sim_i2t = logit_scale * image_features @ text_features_gathered.t()
        sim_t2i = logit_scale * text_features @ image_features_gathered.t()
        bsz = sim_i2t.size(0)

        def reverse_ce(logits: torch.Tensor):
            """Symmetric reverse CE for LR-ITC."""
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
            # fallback to soft targets when gathered batch size differs
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()
            ln_itc = 0.5 * (loss_i2t + loss_t2i)

        lr_i2t = reverse_ce(sim_i2t)
        lr_t2i = reverse_ce(sim_t2i)
        lr_itc = 0.5 * (lr_i2t + lr_t2i)

        itc_loss = ln_itc + lr_itc
        return itc_loss, ln_itc, lr_itc

    def compute_simclr_loss(self, logits_a, logits_b, logits_a_gathered, logits_b_gathered, labels, temperature):
        sim_aa = logits_a @ logits_a_gathered.t() / temperature
        sim_ab = logits_a @ logits_b_gathered.t() / temperature
        sim_ba = logits_b @ logits_a_gathered.t() / temperature
        sim_bb = logits_b @ logits_b_gathered.t() / temperature
        masks = torch.where(F.one_hot(labels, logits_a_gathered.size(0)) == 0, 0, float('-inf'))
        sim_aa += masks
        sim_bb += masks
        sim_a = torch.cat([sim_ab, sim_aa], 1)
        sim_b = torch.cat([sim_ba, sim_bb], 1)
        loss_a = F.cross_entropy(sim_a, labels)
        loss_b = F.cross_entropy(sim_b, labels)
        return (loss_a + loss_b) * 0.5

    def _build_mlp(self, in_dim=512, mlp_dim=512, out_dim=512):
        return nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, out_dim)
        )

    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except:
            try:
                return self.visual.head.weight.dtype
            except:
                try:
                    return self.visual.stem[0].weight.dtype
                except:
                    return self.encode_text.text_projection.weight.dtype

    def encode_image(self, image, cam_id=None, training=False, return_dense=False):
        if return_dense:
            output = self.visual(image.type(self.dtype), cam_id=cam_id, training=training, return_dense=return_dense)
            if self.use_trvd and not training:
                img_feat, dense_feat, attn = output
                if self.use_shared_img_head:
                    m_img = self.shared_head_img(F.normalize(img_feat, dim=-1))
                    m_img = F.normalize(m_img, dim=-1)
                else:
                    m_img = F.normalize(img_feat, dim=-1)
                return m_img, dense_feat, attn
            return output
        output = self.visual(image.type(self.dtype), training=training)
        if self.use_trvd and not training:
            if self.use_shared_img_head:
                m_img = self.shared_head_img(F.normalize(output, dim=-1))
                m_img = F.normalize(m_img, dim=-1)
            else:
                m_img = F.normalize(output, dim=-1)
            return m_img
        return output

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
