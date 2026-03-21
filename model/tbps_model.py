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

        self.visual_emb_layer = VisualEmbeddingLayer(ratio=0.3)
        self.textual_emb_layer = TexualEmbeddingLayer(ratio=0.3)

        # Aerial correction mode
        loss_cfg = getattr(config, "loss", None)
        self.use_aerial_correction = bool(getattr(loss_cfg, "use_aerial_correction", False)) if loss_cfg is not None else False
        if self.use_aerial_correction:
            self.gate_teacher = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Sigmoid(),
            )
            self.gate_student = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Sigmoid(),
            )
            self.view_proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim),
            )

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

        # ----- Aerial correction (train/infer) -----
        corrected_image_features = image_features
        a2t_loss = torch.zeros((), device=image_features.device)
        tre_loss = torch.zeros((), device=image_features.device)
        ice_loss = torch.zeros((), device=image_features.device)
        distill_loss = torch.zeros((), device=image_features.device)
        a2g_loss = torch.zeros((), device=image_features.device)
        shift_loss = torch.zeros((), device=image_features.device)
        reg_loss = torch.zeros((), device=image_features.device)
        shift_acc_fa = 0.0
        shift_acc_fg = 0.0

        if self.use_aerial_correction:
            aerial_mask = cam_id == 0
            ground_mask = cam_id == 1

            if aerial_mask.any():
                f_a = image_features[aerial_mask]
                f_t_a = text_features[aerial_mask]

                # view residual
                v_a = self.view_proj(f_a)
                # student gate (always)
                g_student = self.gate_student(f_a)
                # teacher gate (train only)
                if training:
                    ft_cat = torch.cat([f_a, f_t_a], dim=-1)
                    g_teacher = self.gate_teacher(ft_cat)
                    distill_loss = F.mse_loss(g_student, g_teacher.detach())
                # refined feature
                delta_a = g_student * v_a
                h_a = F.normalize(f_a - 0.1 * delta_a, dim=-1)

                # replace aerial rows in corrected features
                corrected_image_features = image_features.clone()
                corrected_image_features[aerial_mask] = h_a.to(corrected_image_features.dtype)

                # ICE: identity consistency enhancement
                ice_loss = (1 - F.cosine_similarity(h_a, f_a, dim=-1)).mean()

                # a2g_loss is invalid for this dataset (no shared IDs across views)
                a2g_loss = torch.zeros((), device=image_features.device)

        # ----- Global alignment (corrected image features) -----
        image_features_norm = F.normalize(corrected_image_features)
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

        ga_loss, ln_itc, lr_itc = self.calc_contrastive(
            image_features_norm, text_features_norm, image_features_s_norm,
            text_features_s_norm, image_features_norm_gathered,
            text_features_norm_gathered, image_features_s_norm_gathered,
            text_features_s_norm_gathered, sim_targets, alpha, logit_scale
        )

        # ----- Local alignment (unchanged) -----
        i_tse_f = F.normalize(self.visual_emb_layer(feaAttn_i, image_seq_embeddings), dim=-1)
        t_tse_f = F.normalize(self.textual_emb_layer(feaAttn_t, text_tokens, text_seq_embeddings), dim=-1)
        i_tse_f_norm_gathered = self.all_gather(i_tse_f)
        t_tse_f_norm_gathered = self.all_gather(t_tse_f)
        i_tse_f_s_gathered = i_tse_f_norm_gathered.detach()
        t_tse_f_s_gathered = t_tse_f_norm_gathered.detach()
        la_loss, ln_itc_tse, lr_itc_tse = self.calc_contrastive(
            i_tse_f, t_tse_f, i_tse_f.detach(), t_tse_f.detach(),
            i_tse_f_norm_gathered, t_tse_f_norm_gathered,
            i_tse_f_s_gathered, t_tse_f_s_gathered, sim_targets, alpha, logit_scale
        )

        ret['ga_loss'] = ga_loss * self.config.experiment.nitc_ratio
        ret['la_loss'] = la_loss * self.config.experiment.nitc_ratio
        ret['itc_tse_loss'] = ret['la_loss']
        ret['itc_loss'] = ret['ga_loss'] + ret['la_loss']
        ret['ln_itc'] = ln_itc
        ret['lr_itc'] = lr_itc
        ret['ln_itc_tse'] = ln_itc_tse
        ret['lr_itc_tse'] = lr_itc_tse

        # ----- ID loss (corrected image features) -----
        if self.config.experiment.id:
            image_logits = self.classifier(corrected_image_features)
            text_logits = self.classifier(text_features)
            id_loss = (F.cross_entropy(image_logits, ids) + F.cross_entropy(text_logits, ids)) / 2
            ret['id_loss'] = id_loss * self.config.experiment.id_ratio
        else:
            ret['id_loss'] = torch.zeros((), device=image_features.device)

        # ----- Aerial correction losses -----
        ret['ice_loss'] = ice_loss
        ret['distill_loss'] = distill_loss

        # ----- Total loss -----
        if self.use_aerial_correction:
            loss_cfg = getattr(self.config, "loss", None)
            lambda_distill = getattr(loss_cfg, "lambda_distill", 0.5) if loss_cfg is not None else 0.5
            lambda_ice = getattr(loss_cfg, "lambda_ice", 0.05) if loss_cfg is not None else 0.05
            total = ret['ga_loss'] + ret['la_loss'] + ret['id_loss']
            total = total + lambda_distill * ret['distill_loss']
            total = total + lambda_ice * ret['ice_loss']
            ret['total_loss'] = total
        else:
            ret['total_loss'] = ret['itc_loss'] + ret['id_loss']

        return ret

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
            output = self.visual(image.type(self.dtype), cam_id=cam_id, training=training, return_dense=return_dense)
            if isinstance(output, (tuple, list)):
                img_feat, dense_feat, attn = output
            else:
                img_feat, dense_feat, attn = output, None, None
            if self.use_aerial_correction and not training and cam_id is not None:
                corrected = img_feat
                aerial_mask = cam_id == 0
                if aerial_mask.any():
                    f_a = img_feat[aerial_mask]
                    v_a = self.view_proj(f_a)
                    g_student = self.gate_student(f_a)
                    m_a = f_a - g_student * v_a
                    h_a = F.normalize(f_a + 0.1 * m_a, dim=-1)
                    corrected = img_feat.clone()
                    corrected[aerial_mask] = h_a
                return corrected, dense_feat, attn
            return output
        output = self.visual(image.type(self.dtype), training=training)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if self.use_aerial_correction and not training and cam_id is not None:
            corrected = output
            aerial_mask = cam_id == 0
            if aerial_mask.any():
                f_a = output[aerial_mask]
                v_a = self.view_proj(f_a)
                g_student = self.gate_student(f_a)
                m_a = f_a - g_student * v_a
                h_a = F.normalize(f_a + 0.1 * m_a, dim=-1)
                corrected = output.clone()
                corrected[aerial_mask] = h_a
            return corrected
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
