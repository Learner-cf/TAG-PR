## 🔥【AAAI 2026】Text-based Aerial-Ground Person Retrieval

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2511.08369-b31b1b.svg)](https://arxiv.org/abs/2511.08369) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <a href="#-news">News</a> •
  <a href="#-setup">Setup</a> •
  <a href="#-dataset-preparation">Dataset</a> •
  <a href="#-configuration">Config</a> •
  <a href="#-training">Training</a> •
  <a href="#-citation">Citation</a>
</p>

</div>

---

## 📢 News
* **[2025-12]** Code and dataset released
* **[2025-11]** Paper uploaded to arXiv

---

## 🛠️ Setup

### Requirements

*   **Hardware**: 4 × NVIDIA RTX 3090 GPUs (24GB VRAM recommended)
*   **Software**: Python 3.8+ (Recommended), CUDA 11.x+

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/YourUsername/TAG-CLIP.git
cd TAG-CLIP
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation

We provide the processed dataset via Quark Cloud Drive.

1.  **Download**: [Click here to download](https://pan.quark.cn/s/dcddfa17cb7f?pwd=8pE6) (Access Code: `8pE6`)
2.  **Organize**: Extract and arrange the files as follows:

```text
dataset/
├── anno_dir/
│   ├── train_reid.json
│   └── test_reid.json
└── images/
    ├── 0001.jpg
    ├── 0002.jpg
    └── ...
```

---

## ⚙️ Configuration

Modify the configuration file located at `config/s.config.yaml`.

```yaml
# Data Paths
anno_dir: "/path/to/dataset/anno_dir"      # ⚠️ Absolute path to annotation JSONs
image_dir: "/path/to/dataset/images"       # ⚠️ Absolute path to image directory

# Model Settings
model:
  checkpoint: "/path/to/clip/ViT-B-16.pt"  # Path to pre-trained CLIP weights
  # ... other model params
```

---

## 🚀 Training

We support multi-GPU training and shell script execution.

### Option 1: Torchrun 

Use `torchrun` for distributed data parallel (DDP) training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --rdzv_id=12345 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --nnodes=1 \
  --nproc_per_node=4 \
  main.py
```

### Option 2: Shell Script

You can also use the provided shell script wrapper:

```bash
bash shell/train.sh
```

---

## 📝 Citation

If you find this project useful for your research, please consider citing our paper:

```bibtex
@article{zhou2025text,
  title={Text-based Aerial-Ground Person Retrieval},
  author={Zhou, Xinyu and Wu, Yu and Ma, Jiayao and Wang, Wenhao and Cao, Min and Ye, Mang},
  journal={arXiv preprint arXiv:2511.08369},
  year={2025}
}
```

## 📄 License

This project is released under the [MIT License](LICENSE).