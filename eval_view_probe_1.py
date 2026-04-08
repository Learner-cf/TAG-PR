import argparse
import torch
import torch.nn.functional as F

from misc.utils import parse_config
from misc.data import build_pedes_data
from misc.build import load_checkpoint
from model.tbps_model import clip_vitb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    config = parse_config(args.config)
    config.model.checkpoint = "ckpts/checkpoint_best.pth"
    config.model.ckpt_type = "saved"

    dataloader = build_pedes_data(config)
    train_loader = dataloader["train_loader"]
    test_loader = dataloader.get("test_loader", None)

    if train_loader is not None and hasattr(train_loader.dataset, "person2text"):
        num_classes = len(train_loader.dataset.person2text)
    elif test_loader is not None and hasattr(test_loader.dataset, "person2text"):
        num_classes = len(test_loader.dataset.person2text)
    else:
        num_classes = int(getattr(config, "num_classes", 0))
    config.num_classes = num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = clip_vitb(config, num_classes).to(device)
    model, _, _ = load_checkpoint(model, config)
    model.eval()

    correct_cls = 0
    correct_id = 0
    correct_test = 0
    total = 0
    entropy_sum = 0.0
    entropy_sum_test = 0.0
    aerial_total = 0
    aerial_correct_cls = 0
    aerial_correct_id = 0
    aerial_correct_test = 0
    ground_total = 0
    ground_correct_cls = 0
    ground_correct_id = 0
    ground_correct_test = 0

    with torch.no_grad():
        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            if device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    ret = model(batch, training=False)
            else:
                ret = model(batch, training=False)

            f_cls = ret["f_cls"].float()
            f_id = ret["f_id"].float()
            f_test = ret["f_test"].float()
            labels = batch["cam_id"].long()

            logits_cls = model.view_classifier(f_cls)
            logits_id = model.view_classifier(f_id)
            logits_test = model.view_classifier(f_test)

            pred_cls = logits_cls.argmax(dim=1)
            pred_id = logits_id.argmax(dim=1)
            pred_test = logits_test.argmax(dim=1)

            correct_cls += (pred_cls == labels).sum().item()
            correct_id += (pred_id == labels).sum().item()
            correct_test += (pred_test == labels).sum().item()
            total += labels.size(0)

            prob_id = F.softmax(logits_id, dim=1)
            entropy = -(prob_id * (prob_id + 1e-8).log()).sum(dim=1).mean().item()
            entropy_sum += entropy * labels.size(0)

            prob_test = F.softmax(logits_test, dim=1)
            entropy_test = -(prob_test * (prob_test + 1e-8).log()).sum(dim=1).mean().item()
            entropy_sum_test += entropy_test * labels.size(0)

            aerial_mask = (labels == 0)
            aerial_total += aerial_mask.sum().item()
            aerial_correct_cls += ((pred_cls == 0) & aerial_mask).sum().item()
            aerial_correct_id += ((pred_id == 0) & aerial_mask).sum().item()
            aerial_correct_test += ((pred_test == 0) & aerial_mask).sum().item()
            ground_mask = (labels == 1)
            ground_total += ground_mask.sum().item()
            ground_correct_cls += ((pred_cls == 1) & ground_mask).sum().item()
            ground_correct_id += ((pred_id == 1) & ground_mask).sum().item()
            ground_correct_test += ((pred_test == 1) & ground_mask).sum().item()

    acc_fcls = correct_cls / max(1, total)
    acc_fid = correct_id / max(1, total)
    acc_ftest = correct_test / max(1, total)
    avg_entropy = entropy_sum / max(1, total)
    avg_entropy_test = entropy_sum_test / max(1, total)
    aerial_acc_fcls = aerial_correct_cls / max(1, aerial_total)
    aerial_acc_fid = aerial_correct_id / max(1, aerial_total)
    aerial_acc_ftest = aerial_correct_test / max(1, aerial_total)
    ground_acc_fcls = ground_correct_cls / max(1, ground_total)
    ground_acc_fid = ground_correct_id / max(1, ground_total)
    ground_acc_ftest = ground_correct_test / max(1, ground_total)

    print(f"View Acc on f_cls: {acc_fcls:.4f}")
    print(f"View Acc on f_id : {acc_fid:.4f}")
    print(f"View Acc on f_test: {acc_ftest:.4f}")
    print(f"Aerial View Acc on f_cls : {aerial_acc_fcls:.4f}")
    print(f"Aerial View Acc on f_id  : {aerial_acc_fid:.4f}")
    print(f"Aerial View Acc on f_test: {aerial_acc_ftest:.4f}")
    print(f"Ground View Acc on f_cls : {ground_acc_fcls:.4f}")
    print(f"Ground View Acc on f_id  : {ground_acc_fid:.4f}")
    print(f"Ground View Acc on f_test: {ground_acc_ftest:.4f}")
    print(f"Avg entropy on f_id: {avg_entropy:.4f}")
    print(f"Avg entropy on f_test: {avg_entropy_test:.4f}")


if __name__ == "__main__":
    main()
