import random
import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import copy

from misc import utils
from misc.utils import is_using_distributed
from text_utils.tokenizer import tokenize
from .visual_transformer_MoE import visual_transformer
from .text_transformer import text_transformers
from .base_transformer import Transformer, LayerNorm, QuickGELU

from .shared_modules import AllGather
from collections import OrderedDict

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

        if config.experiment.id:
            self.classifier = nn.Linear(self.embed_dim, num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

    def forward(self, input, alpha, training=False):
        ret = dict()

        images = input['image'].to(self.config.device)
        texts = input['caption']
        cam_id = input['cam_id']

        text_tokens = tokenize(texts, context_length=self.config.experiment.text_length).to(self.config.device)
        ids = input['id'].to(self.config.device)

        image_features, image_seq_embeddings, losses, feaAttn_i = self.encode_image(
            images, cam_id=cam_id, return_dense=True, training=training
        )
        text_features, text_seq_embeddings, feaAttn_t = self.encode_text(
            text_tokens, return_dense=True, training=training
        )
        image_features_norm = F.normalize(image_features)
        text_features_norm = F.normalize(text_features)
        image_features_norm_gathered = self.all_gather(image_features_norm)
        text_features_norm_gathered = self.all_gather(text_features_norm)
        view_ratio = getattr(self.config.experiment, 'view_ratio', 1.0)
        ortho_ratio = getattr(self.config.experiment, 'ortho_ratio', 1.0)
        ret['view_loss'] = losses.get('view_loss', 0) * view_ratio
        ret['view_acc'] = losses.get('view_acc', 0)
        ret['ortho_loss'] = losses.get('ortho_loss', 0) * ortho_ratio

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

        nitc_loss = self.calc_contrastive(image_features_norm, text_features_norm, image_features_s_norm,
                                          text_features_s_norm, image_features_norm_gathered,
                                          text_features_norm_gathered, image_features_s_norm_gathered,
                                          text_features_s_norm_gathered, sim_targets, alpha, logit_scale)

        i_tse_f = F.normalize(self.visual_emb_layer(feaAttn_i, image_seq_embeddings), dim=-1)
        t_tse_f = F.normalize(self.textual_emb_layer(feaAttn_t, text_tokens, text_seq_embeddings), dim=-1)
        i_tse_f_norm_gathered = self.all_gather(i_tse_f)
        t_tse_f_norm_gathered = self.all_gather(t_tse_f)
        i_tse_f_s_gathered = i_tse_f_norm_gathered.detach()
        t_tse_f_s_gathered = t_tse_f_norm_gathered.detach()
        nitc_tse_loss = self.calc_contrastive(i_tse_f, t_tse_f, i_tse_f.detach(), t_tse_f.detach(),
                                              i_tse_f_norm_gathered, t_tse_f_norm_gathered,
                                              i_tse_f_s_gathered, t_tse_f_s_gathered, sim_targets, alpha, logit_scale)

        ret['itc_loss'] = (nitc_loss + nitc_tse_loss) * self.config.experiment.nitc_ratio


        if self.config.experiment.ritc:
            ritc_loss = self.compute_ritc(image_features_norm, text_features_norm, image_features_norm_gathered,
                                          text_features_norm_gathered, logit_scale, sim_targets)
            ritc_tse_loss = self.compute_ritc(i_tse_f, t_tse_f, i_tse_f_norm_gathered, t_tse_f_norm_gathered,
                                              logit_scale, sim_targets)
            ret['ritc_loss'] = (ritc_loss + ritc_tse_loss) * self.config.experiment.ritc_ratio

        if self.config.experiment.id:
            image_logits = self.classifier(image_features)
            text_logits = self.classifier(text_features)
            id_loss = (F.cross_entropy(image_logits, ids) + F.cross_entropy(text_logits, ids)) / 2
            ret['id_loss'] = id_loss * self.config.experiment.id_ratio

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
        with torch.no_grad():
            sim_i2t_s = logit_scale * image_features_s @ text_features_s_gathered.t()
            sim_t2i_s = logit_scale * text_features_s @ image_features_s_gathered.t()
            sim_i2t_targets = alpha * F.softmax(sim_i2t_s, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_s, dim=1) + (1 - alpha) * sim_targets  # soft + hard
        sim_i2t = logit_scale * image_features @ text_features_gathered.t()
        sim_t2i = logit_scale * text_features @ image_features_gathered.t()
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        loss_ita = (loss_i2t + loss_t2i) / 2
        return loss_ita

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
            return output
        output = self.visual(image.type(self.dtype), training=training)
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