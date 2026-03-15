import json
import os
import re
from collections import defaultdict

import torch
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def _recover_json_array_prefix(raw_text):
    """Recover valid prefix objects from a possibly truncated JSON array string."""
    text = raw_text.lstrip('\ufeff').strip()
    if not text.startswith('['):
        return None, 0, 0

    decoder = json.JSONDecoder()
    idx = 1
    n = len(text)
    items = []

    while idx < n:
        while idx < n and text[idx] in ' \r\n\t,':
            idx += 1
        if idx >= n:
            break
        if text[idx] == ']':
            return items, idx + 1, n

        try:
            obj, next_idx = decoder.raw_decode(text, idx)
            items.append(obj)
            idx = next_idx
        except json.JSONDecodeError:
            break

    return (items if items else None), idx, n




def infer_domain_label(ann):
    """Deterministically map sample to platform domain: aerial->0, ground->1."""
    fields = [
        ann.get('domain'), ann.get('dataset'), ann.get('source'), ann.get('modality'), ann.get('view')
    ]
    file_path = str(ann.get('file_path', '')).lower()
    joined = ' '.join([str(x).lower() for x in fields if x is not None] + [file_path])

    # aerial-domain hints (PRAI branch in this project includes G2APS-style paths)
    if any(k in joined for k in ['prai', 'prai-1581', 'g2aps', 'aerial', 'uav', 'uavhuman', 'drone', 'ag-reid']):
        return 0
    # ground-domain hints
    if any(k in joined for k in ['market', 'market-1501', 'ground', 'street']):
        return 1

    raise RuntimeError(
        'Cannot infer domain label (aerial/ground) from annotation. '
        'Please include one of [domain/dataset/source/modality/view] or a path hint containing PRAI/Market. '
        f'Problematic sample keys: {list(ann.keys())[:10]}, file_path={ann.get("file_path", "")}'
    )


def load_json_or_raise(json_path):
    """Load JSON with clearer errors; recover valid prefix if file is truncated."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()

        recovered, stop_idx, total_len = _recover_json_array_prefix(raw)
        if recovered is not None and len(recovered) > 0:
            print(
                "[Warning] Recovered from malformed annotation JSON by loading valid prefix only. "
                f"file={json_path}, recovered_items={len(recovered)}, parse_stop_char={stop_idx}/{total_len}."
            )
            print(
                "[Warning] Please repair/regenerate this JSON file for full-data training "
                "(e.g., `python -m json.tool <file>`)."
            )
            return recovered

        msg = (
            f"Invalid JSON in annotation file: {json_path}\n"
            f"JSONDecodeError: {e}\n"
            "Hint: the file may be truncated/corrupted. "
            "Please verify with `python -m json.tool <file>` or regenerate the annotation file."
        )
        raise RuntimeError(msg) from e


class ps_train_dataset(Dataset):
    def __init__(self, ann_root, image_root, transform, split, max_words=30):
        ann_file = os.path.join(ann_root, split + '_reid.json')
        anns = load_json_or_raise(ann_file)
        self.transform = transform
        self.person2text = defaultdict(list)

        person_id2idx = {}
        n = 0
        self.pairs = []

        for ann in anns:
            image_path = os.path.join(image_root, ann['file_path'])
            person_id = ann['id']
            # for single view datasets, we set cam_id to 1
            cam_id = ann.get('cam_id', 1)
            if person_id not in person_id2idx.keys():
                person_id2idx[person_id] = n
                n += 1
            person_idx = person_id2idx[person_id]
            for caption in ann['captions']:
                caption = pre_caption(caption, max_words)
                domain_id = infer_domain_label(ann)
                self.pairs.append((image_path, caption, cam_id, person_idx, domain_id))
                self.person2text[person_idx].append(caption)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, caption, cam_id, person, domain_id = self.pairs[index]
        image_pil = Image.open(image_path)
        image = self.transform(image_pil.convert('RGB'))
        return {
            'image': image,
            'caption': caption,
            'cam_id': cam_id,
            'id': person,
            'domain_id': domain_id,
        }


class ps_eval_dataset(Dataset):
    def __init__(self, ann_root, image_root, transform, split, max_words=30):
        ann_file = os.path.join(ann_root, split + '_reid.json')
        anns = load_json_or_raise(ann_file)
        self.transform = transform
        self.text = []
        self.image = []
        self.txt2person = []
        self.img2person = []

        for ann in anns:
            image_path = os.path.join(image_root, ann['file_path'])
            self.image.append(image_path)

            person_id = ann['id']
            self.img2person.append(person_id)
            for caption in ann['captions']:
                self.text.append(pre_caption(caption, max_words))
                self.txt2person.append(person_id)

        self.txt2person = torch.tensor(self.txt2person, dtype=torch.long)
        self.img2person = torch.tensor(self.img2person, dtype=torch.long)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = self.image[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image


def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption