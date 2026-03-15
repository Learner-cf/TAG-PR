import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from misc.caption_dataset import pre_caption, load_json_or_raise, infer_domain_label
from misc.eval import test_tse
from misc.utils import parse_config, set_seed
from model.tbps_model_MoE import clip_vitb
from text_utils.tokenizer import tokenize


def infer_domain_from_ann_file(ann_file):
    name = os.path.basename(str(ann_file)).lower()
    if "prai" in name:
        return 0
    if "market" in name:
        return 1
    return None


class DomainSupervisedTrainDataset(Dataset):
    """Stage1 clean domain-supervised set built from PRAI + Market annotations."""

    def __init__(self, ann_files, image_root, transform, max_words=77):
        self.transform = transform
        self.image_root = image_root
        self.samples = []
        self.person2idx = {}

        for ann_file in ann_files:
            anns = load_json_or_raise(ann_file)
            file_level_domain = infer_domain_from_ann_file(ann_file)
            for ann in anns:
                file_path = ann.get('file_path') or ann.get('image')
                if not file_path:
                    continue
                domain_id = file_level_domain if file_level_domain is not None else infer_domain_label(ann)
                if domain_id not in (0, 1):
                    raise RuntimeError(f"Invalid domain_id={domain_id} for ann_file={ann_file}")
                cam_id = ann.get('cam_id', 1)
                source_tag = 'prai' if domain_id == 0 else 'market'
                raw_pid = f"{source_tag}:{ann.get('id')}"
                if raw_pid not in self.person2idx:
                    self.person2idx[raw_pid] = len(self.person2idx)
                person_idx = self.person2idx[raw_pid]

                captions = ann.get('captions') or ann.get('caption') or []
                if isinstance(captions, str):
                    captions = [captions]
                for cap in captions:
                    self.samples.append({
                        'file_path': file_path,
                        'caption': pre_caption(str(cap), max_words),
                        'cam_id': cam_id,
                        'id': person_idx,
                        'domain_id': domain_id,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        full = s['file_path'] if os.path.isabs(s['file_path']) else os.path.join(self.image_root, s['file_path'])
        image = self.transform(Image.open(full).convert('RGB'))
        return {
            'index': index,
            'image': image,
            'caption': s['caption'],
            'cam_id': s['cam_id'],
            'id': s['id'],
            'domain_id': s['domain_id'],
        }


class MixedTrainDataset(DomainSupervisedTrainDataset):
    """Stage2 mixed train set from train_reid.json with pseudo domain labels."""
    # Intentionally reuses parent __getitem__; this subclass customizes only annotation parsing.

    def __init__(self, ann_file, image_root, transform, max_words=77):
        self.transform = transform
        self.image_root = image_root
        self.samples = []
        self.person2idx = {}

        anns = load_json_or_raise(ann_file)
        for ann in anns:
            file_path = ann.get('file_path') or ann.get('image')
            if not file_path:
                continue
            cam_id = ann.get('cam_id', 1)
            raw_pid = str(ann.get('id'))
            if raw_pid not in self.person2idx:
                self.person2idx[raw_pid] = len(self.person2idx)
            person_idx = self.person2idx[raw_pid]

            captions = ann.get('captions') or ann.get('caption') or []
            if isinstance(captions, str):
                captions = [captions]
            for cap in captions:
                self.samples.append({
                    'file_path': file_path,
                    'caption': pre_caption(str(cap), max_words),
                    'cam_id': cam_id,
                    'id': person_idx,
                    'domain_id': -1,
                })


class CleanEvalDataset(Dataset):
    """Retrieval-style eval dataset built from PRAI + market files for stage1 clean validation."""

    def __init__(self, ann_files, image_root, transform, max_words=77):
        self.transform = transform
        self.text = []
        self.image = []
        self.txt2person = []
        self.img2person = []
        self.person2idx = {}

        for ann_file in ann_files:
            anns = load_json_or_raise(ann_file)
            file_level_domain = infer_domain_from_ann_file(ann_file)
            for ann in anns:
                file_path = ann.get('file_path') or ann.get('image')
                if not file_path:
                    continue
                domain_id = file_level_domain if file_level_domain is not None else infer_domain_label(ann)
                if domain_id not in (0, 1):
                    raise RuntimeError(f"Invalid domain_id={domain_id} for ann_file={ann_file}")
                source_tag = 'prai' if domain_id == 0 else 'market'
                raw_pid = f"{source_tag}:{ann.get('id')}"
                if raw_pid not in self.person2idx:
                    self.person2idx[raw_pid] = len(self.person2idx)
                person_idx = self.person2idx[raw_pid]

                full = file_path if os.path.isabs(file_path) else os.path.join(image_root, file_path)
                self.image.append(full)
                self.img2person.append(person_idx)

                captions = ann.get('captions') or ann.get('caption') or []
                if isinstance(captions, str):
                    captions = [captions]
                for cap in captions:
                    self.text.append(pre_caption(str(cap), max_words))
                    self.txt2person.append(person_idx)

        self.txt2person = torch.tensor(self.txt2person, dtype=torch.long)
        self.img2person = torch.tensor(self.img2person, dtype=torch.long)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image = Image.open(self.image[index]).convert('RGB')
        return self.transform(image)


def build_transform(size, train=True):
    if isinstance(size, int):
        size = (size, size)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        normalize,
    ])


def build_test_loader(config):
    from misc.caption_dataset import ps_eval_dataset

    val_tf = build_transform(config.experiment.input_resolution, train=False)
    test_dataset = ps_eval_dataset(config.anno_dir, config.image_dir, val_tf, split='test', max_words=77)
    return DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=config.data.num_workers, drop_last=False)


def build_clean_val_loader(config, clean_anns):
    val_tf = build_transform(config.experiment.input_resolution, train=False)
    ds = CleanEvalDataset(clean_anns, config.image_dir, val_tf, max_words=77)
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=config.data.num_workers, drop_last=False)


def pairwise_info_nce(a, b, ids, temp=0.07):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = (a @ b.t()) / temp
    same = (ids.view(-1, 1) == ids.view(1, -1)).float()
    targets = same / same.sum(dim=1, keepdim=True).clamp_min(1.0)
    return -(F.log_softmax(logits, dim=1) * targets).sum(dim=1).mean()


def batch_hard_triplet(feat, ids, margin=0.3):
    feat = F.normalize(feat, dim=-1)
    d = torch.cdist(feat, feat, p=2)
    same = ids[:, None] == ids[None, :]
    eye = torch.eye(len(ids), device=ids.device, dtype=torch.bool)
    pos_mask = same & ~eye
    neg_mask = ~same

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.zeros((), device=ids.device)

    hardest_pos = torch.where(pos_mask, d, torch.full_like(d, -1e9)).max(dim=1).values
    hardest_neg = torch.where(neg_mask, d, torch.full_like(d, 1e9)).min(dim=1).values
    loss = F.relu(hardest_pos - hardest_neg + margin)
    valid = (hardest_pos > -1e8) & (hardest_neg < 1e8)
    if valid.sum() == 0:
        return torch.zeros((), device=ids.device)
    return loss[valid].mean()




def _looks_like_state_dict(d):
    return isinstance(d, dict) and len(d) > 0 and all(isinstance(v, torch.Tensor) for v in d.values())


def extract_model_state_dict(ckpt):
    if isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        return ckpt['model']
    if _looks_like_state_dict(ckpt):
        return ckpt
    raise RuntimeError(
        'Invalid checkpoint for model loading. Expected either a full checkpoint with key "model" '
        'or a pure model state_dict. Stage 2 requires a Stage 1 checkpoint with model weights.'
    )


def extract_domain_head_state_dict(ckpt):
    if isinstance(ckpt, dict) and 'domain_head' in ckpt and isinstance(ckpt['domain_head'], dict):
        return ckpt['domain_head']
    return None


def infer_num_classes_from_ckpt(ckpt_model_state, fallback):
    w = ckpt_model_state.get('classifier.weight') if isinstance(ckpt_model_state, dict) else None
    if isinstance(w, torch.Tensor) and w.ndim == 2 and w.shape[0] > 0:
        return int(w.shape[0])
    return fallback


def load_state_flexible(model, state_dict, tag='model'):
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k not in model_state:
            continue
        if model_state[k].shape != v.shape:
            skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        filtered[k] = v
    ret = model.load_state_dict(filtered, strict=False)
    print(f"[{tag}] missing={len(ret.missing_keys)} unexpected={len(ret.unexpected_keys)} skipped_shape={len(skipped)}")
    for k, s1, s2 in skipped[:10]:
        print(f"[{tag}] skipped: {k} ckpt{s1} -> model{s2}")


@torch.no_grad()
def generate_pseudo_domain_labels(model, domain_head, dataset, device, batch_size=128):
    model.eval()
    domain_head.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    n = len(dataset)
    pseudo = torch.zeros(n, dtype=torch.long)
    conf = torch.zeros(n, dtype=torch.float)

    for batch in loader:
        idx = batch['index']
        images = batch['image'].to(device)
        cam_id = batch['cam_id'].to(device)

        _, img_dense, _, attn_i = model.encode_image(images, cam_id=cam_id, return_dense=True, training=False)
        img_private = model.visual_emb_layer(attn_i, img_dense, training=False)
        prob = torch.softmax(domain_head(img_private), dim=-1)
        c, y = prob.max(dim=-1)

        pseudo[idx] = y.cpu()
        conf[idx] = c.cpu()

    return pseudo, conf


def compute_weighted_domain_loss(logits, labels, conf, high=0.9, low=0.6):
    ce = F.cross_entropy(logits, labels, reduction='none')
    w = torch.where(conf >= high, torch.ones_like(conf), torch.where(conf >= low, conf, torch.zeros_like(conf)))
    if w.sum() <= 0:
        return torch.zeros((), device=logits.device), 0.0, conf.mean().item()
    loss = (w * ce).sum() / (w.sum() + 1e-6)
    used_ratio = (w > 0).float().mean().item()
    return loss, used_ratio, conf.mean().item()


def train_epoch_decomp(model, domain_head, loader, optimizer, device, config, lambdas, pseudo_pack=None,
                       pseudo_high=0.9, pseudo_low=0.6):
    model.train()
    domain_head.train()
    meter = {k: 0.0 for k in ["L_total", "L_shared_align", "L_id", "L_tri", "L_dom", "L_decouple", "pseudo_used_ratio", "pseudo_conf_mean"]}

    pseudo_labels = pseudo_pack[0] if pseudo_pack is not None else None
    pseudo_conf = pseudo_pack[1] if pseudo_pack is not None else None

    for batch in loader:
        images = batch['image'].to(device)
        texts = batch['caption']
        ids = batch['id'].to(device)
        cam_id = batch['cam_id'].to(device)
        text_tokens = tokenize(texts, context_length=config.experiment.text_length).to(device)

        img_shared, img_dense, _, attn_i = model.encode_image(images, cam_id=cam_id, return_dense=True, training=True)
        txt_shared, _, _ = model.encode_text(text_tokens, return_dense=True, training=True)
        img_private = model.visual_emb_layer(attn_i, img_dense, training=True)

        l_shared_align = pairwise_info_nce(img_shared, txt_shared, ids)

        l_id = torch.zeros((), device=device)
        if hasattr(model, 'classifier'):
            l_id = 0.5 * (F.cross_entropy(model.classifier(img_shared), ids) + F.cross_entropy(model.classifier(txt_shared), ids))

        all_shared = torch.cat([img_shared, txt_shared], dim=0)
        all_ids = torch.cat([ids, ids], dim=0)
        l_tri = batch_hard_triplet(all_shared, all_ids, margin=lambdas['tri_margin'])

        dom_logits = domain_head(img_private)
        if pseudo_labels is None:
            dom_label = batch['domain_id'].to(device).long()
            l_dom = F.cross_entropy(dom_logits, dom_label)
            used_ratio = 1.0
            conf_mean = 1.0
        else:
            idx = batch['index'].long()
            dom_label = pseudo_labels[idx].to(device)
            conf = pseudo_conf[idx].to(device)
            l_dom, used_ratio, conf_mean = compute_weighted_domain_loss(dom_logits, dom_label, conf, high=pseudo_high, low=pseudo_low)

        s_img = F.normalize(img_shared, dim=-1)
        p_img = F.normalize(img_private, dim=-1)
        l_decouple = ((s_img * p_img).sum(dim=1) ** 2).mean()

        loss = lambdas['align'] * l_shared_align + lambdas['id'] * l_id + lambdas['tri'] * l_tri + lambdas['dom'] * l_dom + lambdas['decouple'] * l_decouple

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meter['L_total'] += loss.item()
        meter['L_shared_align'] += l_shared_align.item()
        meter['L_id'] += l_id.item()
        meter['L_tri'] += l_tri.item()
        meter['L_dom'] += l_dom.item()
        meter['L_decouple'] += l_decouple.item()
        meter['pseudo_used_ratio'] += used_ratio
        meter['pseudo_conf_mean'] += conf_mean

    n = max(1, len(loader))
    return {k: v / n for k, v in meter.items()}


def train_stage0_probe(model, domain_head, train_set, device, epochs, lr, log_dir, batch_size):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # Split by image path (not caption-sample) to reduce leakage between train/val.
    path_to_indices = {}
    for i, smp in enumerate(train_set.samples):
        path_to_indices.setdefault(smp['file_path'], []).append(i)
    unique_paths = sorted(path_to_indices.keys())
    rng = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(unique_paths), generator=rng).tolist()
    n_val_paths = max(1, int(0.2 * len(unique_paths)))
    val_path_set = set(unique_paths[j] for j in perm[:n_val_paths])

    train_indices, val_indices = [], []
    for pth, idxs in path_to_indices.items():
        if pth in val_path_set:
            val_indices.extend(idxs)
        else:
            train_indices.extend(idxs)

    train_loader = DataLoader(torch.utils.data.Subset(train_set, train_indices), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(torch.utils.data.Subset(train_set, val_indices), batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    optimizer = torch.optim.AdamW(domain_head.parameters(), lr=lr)
    log_csv = os.path.join(log_dir, 'training_log_stage0.csv')
    best_acc = -1.0

    with open(log_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'domain_acc', 'best_domain_acc'])
        writer.writeheader()

        for epoch in range(epochs):
            domain_head.train()
            loss_sum = 0.0
            for batch in train_loader:
                images = batch['image'].to(device)
                cam_id = batch['cam_id'].to(device)
                dom = batch['domain_id'].to(device).long()

                with torch.no_grad():
                    _, img_dense, _, attn_i = model.encode_image(images, cam_id=cam_id, return_dense=True, training=False)
                    img_private = model.visual_emb_layer(attn_i, img_dense, training=False)

                logits = domain_head(img_private)
                loss = F.cross_entropy(logits, dom)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            domain_head.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    cam_id = batch['cam_id'].to(device)
                    dom = batch['domain_id'].to(device).long()
                    _, img_dense, _, attn_i = model.encode_image(images, cam_id=cam_id, return_dense=True, training=False)
                    img_private = model.visual_emb_layer(attn_i, img_dense, training=False)
                    pred = domain_head(img_private).argmax(dim=-1)
                    correct += (pred == dom).sum().item()
                    total += dom.numel()

            acc = correct / max(1, total)
            is_best = acc > best_acc
            if is_best:
                best_acc = acc
                torch.save({'domain_head': domain_head.state_dict(), 'epoch': epoch + 1, 'best_domain_acc': best_acc},
                           os.path.join(log_dir, 'stage0_domain_probe_best.pt'))

            row = {
                'epoch': epoch + 1,
                'train_loss': loss_sum / max(1, len(train_loader)),
                'domain_acc': acc,
                'best_domain_acc': best_acc,
            }
            writer.writerow(row)
            f.flush()
            print(row)

    torch.save({'domain_head': domain_head.state_dict(), 'epoch': epochs, 'best_domain_acc': best_acc},
               os.path.join(log_dir, 'stage0_domain_probe_last.pt'))


def save_stage_ckpt(path, model, domain_head, epoch, config, best_metric=None):
    obj = {'model': model.state_dict(), 'domain_head': domain_head.state_dict(), 'epoch': epoch, 'config': config}
    if best_metric is not None:
        obj['best_metric'] = best_metric
    torch.save(obj, path)


def main():
    parser = argparse.ArgumentParser('TAG-PR 3-stage curriculum trainer')
    parser.add_argument('--config', type=str, default='configs/l1_only_config.yaml')
    parser.add_argument('--stage', type=int, default=-1, choices=[-1, 0, 1, 2], help='training stage; -1 means read from config.train_stage')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--device', type=str, default='cuda', help='training device')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--extract_anno', type=str, default='')
    parser.add_argument('--align_lambda', type=float, default=1.0, help='weight for shared text-image alignment loss')
    parser.add_argument('--id_lambda', type=float, default=0.5, help='weight for identity loss on shared branch')
    parser.add_argument('--tri_lambda', type=float, default=1.0, help='weight for triplet loss on shared branch')
    parser.add_argument('--dom_lambda', type=float, default=1.0, help='weight for domain/platform classification loss')
    parser.add_argument('--decouple_lambda', type=float, default=1.0, help='weight for shared-private orthogonality loss')
    parser.add_argument('--tri_margin', type=float, default=0.3)
    parser.add_argument('--resume_stage1', type=str, default='', help='stage1 checkpoint path for stage2 init')
    parser.add_argument('--pseudo_high_thresh', type=float, default=0.9)
    parser.add_argument('--pseudo_low_thresh', type=float, default=0.6)
    parser.add_argument('--stage2_refresh_every', type=int, default=0, help='refresh pseudo labels every N epochs; 0 means fixed labels')
    parser.add_argument('--stage1_clean_val', type=int, default=1, help='1 to validate stage1 on PRAI+Market clean eval set')
    args = parser.parse_args()

    config = parse_config(args.config)
    stage = args.stage if args.stage != -1 else int(getattr(config, 'train_stage', 1))
    config.train_stage = stage
    config.experiment.ritc = False
    config.experiment.view_ratio = 0.0
    config.experiment.ortho_ratio = 0.0
    config.schedule.epoch = args.epochs
    set_seed(config)

    device = torch.device(args.device)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    train_tf = build_transform(config.experiment.input_resolution, train=True)
    clean_train_anns = [
        os.path.join(config.anno_dir, 'PRAI_train.json'),
        os.path.join(config.anno_dir, 'market_train.json'),
    ]
    clean_test_anns = [
        os.path.join(config.anno_dir, 'PRAI_test.json'),
        os.path.join(config.anno_dir, 'market_test.json'),
    ]
    clean_set = DomainSupervisedTrainDataset(clean_train_anns, config.image_dir, train_tf, max_words=77)
    val_loader = build_clean_val_loader(config, clean_test_anns) if (stage == 1 and bool(args.stage1_clean_val)) else build_test_loader(config)

    if stage == 0:
        model = clip_vitb(config, num_classes=max(1, len(clean_set.person2idx))).to(device)
        domain_head = nn.Linear(config.model.embed_dim, 2).to(device)
        train_stage0_probe(model, domain_head, clean_set, device, args.epochs, args.lr, args.log_dir, args.batch_size)
        return

    train_set = clean_set if stage == 1 else MixedTrainDataset(
        os.path.join(config.anno_dir, 'train_reid.json'), config.image_dir, train_tf, max_words=77
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=config.data.num_workers, drop_last=True)

    num_classes = len(train_set.person2idx)
    model = clip_vitb(config, num_classes=max(1, num_classes)).to(device)
    domain_head = nn.Linear(config.model.embed_dim, 2).to(device)

    pseudo_pack = None
    pseudo_file = os.path.join(args.log_dir, 'stage2_pseudo_labels.pt')
    if stage == 2:
        if not args.resume_stage1:
            raise RuntimeError('Stage 2 requires --resume_stage1 path to stage1_best.pt')
        ckpt = torch.load(args.resume_stage1, map_location='cpu')
        ckpt_model = extract_model_state_dict(ckpt)
        ckpt_domain = extract_domain_head_state_dict(ckpt)
        if ckpt_domain is None:
            raise RuntimeError('Stage 2 requires a Stage 1 checkpoint containing both model and domain_head')

        # student: build once using current stage2 class size, then flex-load stage1 weights
        load_state_flexible(model, ckpt_model, tag='stage2-student')
        load_state_flexible(domain_head, ckpt_domain, tag='stage2-domain-head')

        # teacher: frozen stage1 model for stable pseudo labels
        teacher_num_classes = infer_num_classes_from_ckpt(ckpt_model, fallback=max(1, num_classes))
        teacher_model = clip_vitb(config, num_classes=teacher_num_classes).to(device)
        teacher_head = nn.Linear(config.model.embed_dim, 2).to(device)
        load_state_flexible(teacher_model, ckpt_model, tag='stage2-teacher')
        load_state_flexible(teacher_head, ckpt_domain, tag='stage2-teacher-head')
        teacher_model.eval()
        teacher_head.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        for p in teacher_head.parameters():
            p.requires_grad = False

        pseudo_pack = generate_pseudo_domain_labels(teacher_model, teacher_head, train_set, device, batch_size=min(128, args.batch_size))
        torch.save({'epoch': 0, 'pseudo_labels': pseudo_pack[0], 'pseudo_conf': pseudo_pack[1]}, pseudo_file)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(domain_head.parameters()),
                                  lr=args.lr, weight_decay=config.schedule.weight_decay)

    lambdas = {
        'align': args.align_lambda,
        'id': args.id_lambda,
        'tri': args.tri_lambda,
        'dom': args.dom_lambda,
        'decouple': args.decouple_lambda,
        'tri_margin': args.tri_margin,
    }

    stage_name = f'stage{stage}'
    log_fields = ['epoch', 'L_total', 'L_shared_align', 'L_id', 'L_tri', 'L_dom', 'L_decouple']
    if stage == 2:
        log_fields += ['pseudo_conf_used_ratio', 'pseudo_conf_mean']
    log_fields += ['val_r1', 'val_r5', 'val_r10', 'val_map']

    log_csv = os.path.join(args.log_dir, f'training_log_{stage_name}.csv')
    best_path = os.path.join(args.log_dir, f'{stage_name}_best.pt')
    last_path = os.path.join(args.log_dir, f'{stage_name}_last.pt')
    best_r1 = -1.0

    with open(log_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

        for epoch in range(args.epochs):
            if stage == 2 and args.stage2_refresh_every > 0 and epoch > 0 and epoch % args.stage2_refresh_every == 0:
                # optional refresh with frozen teacher (still stable, not student self-bootstrapping)
                pseudo_pack = generate_pseudo_domain_labels(teacher_model, teacher_head, train_set, device, batch_size=min(128, args.batch_size))
                torch.save({'epoch': epoch + 1, 'pseudo_labels': pseudo_pack[0], 'pseudo_conf': pseudo_pack[1]}, pseudo_file)

            tr = train_epoch_decomp(
                model, domain_head, train_loader, optimizer, device, config, lambdas,
                pseudo_pack=pseudo_pack, pseudo_high=args.pseudo_high_thresh, pseudo_low=args.pseudo_low_thresh
            )
            ev = test_tse(model, val_loader, config.experiment.text_length, device)

            row = {
                'epoch': epoch + 1,
                'L_total': tr['L_total'],
                'L_shared_align': tr['L_shared_align'],
                'L_id': tr['L_id'],
                'L_tri': tr['L_tri'],
                'L_dom': tr['L_dom'],
                'L_decouple': tr['L_decouple'],
                'val_r1': ev['r1'],
                'val_r5': ev['r5'],
                'val_r10': ev['r10'],
                'val_map': ev['mAP'],
            }
            if stage == 2:
                row['pseudo_conf_used_ratio'] = tr['pseudo_used_ratio']
                row['pseudo_conf_mean'] = tr['pseudo_conf_mean']

            writer.writerow(row)
            f.flush()
            print(row)

            if ev['r1'] > best_r1:
                best_r1 = ev['r1']
                save_stage_ckpt(best_path, model, domain_head, epoch + 1, config, best_metric=best_r1)

            save_stage_ckpt(last_path, model, domain_head, epoch + 1, config, best_metric=best_r1)

    if args.extract_anno:
        from tools.extract_l1_embeddings import extract_embeddings

        class _Args:
            config = args.config
            ckpt = best_path
            anno = args.extract_anno
            output_dir = args.log_dir
            batch_size = 256
            device = args.device
            image_dir = config.image_dir
            expect_l1_only = True

        extract_embeddings(_Args)


if __name__ == '__main__':
    main()
