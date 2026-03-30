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
from .caption_guided_vdfe import CaptionGuidedAGViewDecoupler


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

        self.use_ag_decoupling = bool(getattr(model_cfg, "use_ag_decoupling", False)) if model_cfg is not None else False
        self.lambda_rm = float(getattr(weights_cfg, "lambda_rm", 0.2)) if weights_cfg is not None else 0.2
        self.lambda_l1 = float(getattr(weights_cfg, "lambda_l1", 0.1)) if weights_cfg is not None else 0.1
        self.lambda_ag = float(getattr(weights_cfg, "lambda_ag", 0.5)) if weights_cfg is not None else 0.5
        self.lambda_v = float(getattr(weights_cfg, "lambda_v", 1.0)) if weights_cfg is not None else 1.0
        self.lambda_inv = float(getattr(weights_cfg, "lambda_inv", 1.0)) if weights_cfg is not None else 1.0
        self.lambda_o = float(getattr(weights_cfg, "lambda_o", 0.1)) if weights_cfg is not None else 0.1
        self.ag_num_classes = int(getattr(model_cfg, "ag_num_classes", 2)) if model_cfg is not None else 2
        self.ag_dropout = float(getattr(model_cfg, "ag_dropout", 0.0)) if model_cfg is not None else 0.0

        if self.use_ag_decoupling:
            self.ag_decoupler = CaptionGuidedAGViewDecoupler(
                embed_dim=self.embed_dim,
                num_heads=int(getattr(model_cfg, "ag_heads", 8)) if model_cfg is not None else 8,
                dropout=self.ag_dropout,
                lambda_rm=self.lambda_rm,
            )
            self.ag_classifier = nn.Linear(self.embed_dim, self.ag_num_classes)
            nn.init.normal_(self.ag_classifier.weight.data, std=0.001)
            nn.init.constant_(self.ag_classifier.bias.data, val=0.0)
        else:
            self.ag_decoupler = None
            self.ag_classifier = None

        self.visual_emb_layer = VisualEmbeddingLayer(ratio=0.3)
        self.textual_emb_layer = TexualEmbeddingLayer(ratio=0.3)

        if self.use_id_loss:
            self.classifier = nn.Linear(self.embed_dim, num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)
        else:
            self.classifier = None


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

        f_cls_raw = image_features
        f_cls = f_cls_raw
        f_id = None
        f_view = None
        f_hat = None

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
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logit_scale.data = torch.clamp(logit_scale.data, max=100)

        if self.use_global_align and text_features is not None and sim_targets is not None and logit_scale is not None:
            image_features_norm = F.normalize(f_cls)
            text_features_norm = F.normalize(text_features)
            image_features_norm_gathered = self.all_gather(image_features_norm)
            text_features_norm_gathered = self.all_gather(text_features_norm)

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
            ret['ga_loss'] = ga_loss * self.config.experiment.nitc_ratio
            ret['ln_itc'] = ln_itc
            ret['lr_itc'] = lr_itc

        i_tse_f = None
        t_tse_f = None
        if self.use_local_align and feaAttn_i is not None and feaAttn_t is not None and sim_targets is not None and logit_scale is not None:
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
            ret['la_loss'] = la_loss * self.config.experiment.nitc_ratio
            ret['ln_itc_tse'] = ln_itc_tse
            ret['lr_itc_tse'] = lr_itc_tse

        id_loss = torch.zeros((), device=f_cls.device)
        image_logits = None
        text_logits = None
        if self.use_id_loss and self.classifier is not None and text_features is not None and ids is not None:
            image_logits = self.classifier(f_cls)
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

        l1_loss = torch.zeros((), device=f_cls.device)
        ag_dec_loss = torch.zeros((), device=f_cls.device)
        orth_loss = torch.zeros((), device=f_cls.device)
        ag_logits_raw = None
        ag_logits_view = None
        ag_logits_hat = None
        if self.use_ag_decoupling and self.ag_decoupler is not None and text_seq_embeddings is not None:
            f_id, f_view, f_hat = self.ag_decoupler(f_cls_raw, text_seq_embeddings)
            l1_loss = torch.mean(torch.abs(f_hat - f_id.detach()))

            ag_label = None
            for k in ("cam_id", "view", "ag_label"):
                if k in input and input[k] is not None:
                    ag_label = input[k]
                    break
            if ag_label is not None and self.ag_classifier is not None:
                ag_label = ag_label.to(self.config.device).long()
                ag_logits_raw = self.ag_classifier(f_cls_raw)
                ag_logits_view = self.ag_classifier(f_view)
                ag_logits_hat = self.ag_classifier(f_hat)

                loss_raw = F.cross_entropy(ag_logits_raw, ag_label)
                log_p_hat = F.log_softmax(ag_logits_hat, dim=-1)
                uniform = torch.full_like(log_p_hat, 1.0 / log_p_hat.size(1))
                loss_inv = -(uniform * log_p_hat).sum(dim=1).mean()

                orth_loss = torch.abs(F.cosine_similarity(f_id, f_view, dim=-1)).mean() * 100
                ag_dec_loss = loss_raw + self.lambda_inv * loss_inv

                ret['ag_logits_raw'] = ag_logits_raw
                ret['ag_logits_view'] = ag_logits_view
                ret['ag_logits_hat'] = ag_logits_hat
            else:
                orth_loss = torch.abs(F.cosine_similarity(f_id, f_view, dim=-1)).mean() * 100
                ag_dec_loss = torch.zeros((), device=f_cls.device)

            total = total + self.lambda_l1 * l1_loss + self.lambda_ag * ag_dec_loss + self.lambda_o * orth_loss

        ret['l1_loss'] = l1_loss
        ret['ag_dec_loss'] = ag_dec_loss
        ret['orth_loss'] = orth_loss
        if f_id is not None:
            ret['f_id'] = f_id
        if f_view is not None:
            ret['f_view'] = f_view
        if f_hat is not None:
            ret['f_hat'] = f_hat

        ret['total_loss'] = total
        ret['f_cls'] = f_cls
        if text_features is not None:
            ret['t_eos'] = text_features
        if i_tse_f is not None:
            ret['local_img_feat'] = i_tse_f
        if t_tse_f is not None:
            ret['local_txt_feat'] = t_tse_f

        return ret

    def calc_contrastive(self, image_features, text_features, image_features_s, text_features_s,
                         image_features_gathered, text_features_gathered, image_features_s_gathered,
                         text_features_s_gathered,
                         sim_targets, alpha, logit_scale):
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
