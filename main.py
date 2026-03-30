import os
import random
import time
from pathlib import Path
import argparse
import torch
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
from misc.utils import get_rank
from misc.build import load_checkpoint, cosine_scheduler, build_optimizer
from misc.data import build_pedes_data
from misc.eval import test_tse, test
from misc.utils import parse_config, init_distributed_mode, set_seed, is_master, is_using_distributed, \
    AverageMeter
from model.tbps_model import clip_vitb
# from eva_clip import create_model_and_transforms
from text_utils.logger import setup_logger


def _add_meter(meters, name):
    meters[name] = AverageMeter()


def run(config):
    logger = setup_logger('TAG', distributed_rank=get_rank(), save_dir=config.model.saved_path)
    logger.propagate = False
    logger.info(f'\n{config}')
    if getattr(config.model, "cg_vdfe", None) is not None and bool(getattr(config.model.cg_vdfe, "enable", False)):
        logger.info("Running CG-VDFE + L_view training")
    else:
        logger.info("Running baseline (global/local alignment + id loss)")

    # data
    dataloader = build_pedes_data(config)
    train_loader = dataloader['train_loader']
    num_classes = len(train_loader.dataset.person2text)
    config.num_classes = num_classes

    meters = {}
    best_rank_1 = 0.0
    best_epoch = 0

    # model
    # model_name = "EVA02-CLIP-B-16"
    # pretrained = "/data/zxy/UAV/checkpoints/EVA02_CLIP_B_psz16_s8B.pt"  # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"
    # model, _, preprocess = create_model_and_transforms(config, model_name, pretrained, force_custom_clip=True)
    model = clip_vitb(config, num_classes)
    model.to(config.device)
    model, load_result, ckpt = load_checkpoint(model, config)

    if is_using_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.device],
                                                          find_unused_parameters=True)

    # schedule
    config.schedule.niter_per_ep = len(train_loader)
    lr_schedule = cosine_scheduler(config)

    # optimizer
    optimizer = build_optimizer(config, model)

    # train
    it = 0
    start_epoch = 0
    scaler = torch.cuda.amp.GradScaler()

    if ckpt is not None and getattr(config.model, "resume", False):
        loaded_opt = False
        if 'optimizer' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
                loaded_opt = True
            except ValueError:
                logger.warning("Optimizer state incompatible with current model; skipping optimizer resume.")
        if loaded_opt and 'scaler' in ckpt:
            try:
                scaler.load_state_dict(ckpt['scaler'])
            except Exception:
                logger.warning("Scaler state incompatible; skipping scaler resume.")
        if loaded_opt and 'epoch' in ckpt:
            start_epoch = int(ckpt['epoch']) + 1
        if loaded_opt and 'it' in ckpt:
            it = int(ckpt['it'])
        elif loaded_opt:
            it = start_epoch * config.schedule.niter_per_ep
    for epoch in range(start_epoch, config.schedule.epoch):
        if is_using_distributed():
            dataloader['train_sampler'].set_epoch(epoch)

        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        loader_iter = train_loader
        if tqdm is not None:
            loader_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.schedule.epoch}", leave=False)
        for i, batch in enumerate(loader_iter):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[it] * param_group['ratio']

            if epoch == 0:
                alpha = config.model.softlabel_ratio * min(1.0, i / len(train_loader))
            else:
                alpha = config.model.softlabel_ratio

            with torch.autocast(device_type='cuda'):
                ret = model(batch, alpha, training=True)
                loss = ret.get('total_loss', None)
                if loss is None:
                    loss = sum([v for k, v in ret.items() if "loss" in k])

            def _mval(x):
                if torch.is_tensor(x):
                    return x.detach().item()
                return x

            batch_size = batch['image'].shape[0]
            for k, v in ret.items():
                if "loss" not in k and k not in ("gate_mean", "gate_std"):
                    continue
                if k not in meters:
                    _add_meter(meters, k)
                meters[k].update(_mval(v), batch_size)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()
            optimizer.zero_grad()
            it += 1

            if (i + 1) % config.log.print_period == 0:
                log_parts = [f"Epoch[{epoch + 1}] Iteration[{i + 1}/{len(train_loader)}]"]
                if "total_loss" in meters and meters["total_loss"].count > 0:
                    log_parts.append(f"total_loss: {meters['total_loss'].val:.4f}")
                other_losses = [k for k in meters.keys() if k != "total_loss" and k not in ("gate_mean", "gate_std")]
                for k in sorted(other_losses):
                    if meters[k].count > 0:
                        log_parts.append(f"{k}: {meters[k].val:.4f}")
                if "gate_mean" in meters and meters["gate_mean"].count > 0:
                    log_parts.append(f"gate_mean: {meters['gate_mean'].val:.4f}")
                if "gate_std" in meters and meters["gate_std"].count > 0:
                    log_parts.append(f"gate_std: {meters['gate_std'].val:.4f}")
                log_parts.append(f"Base Lr: {param_group['lr']:.2e}")
                log_msg = ", ".join(log_parts)
                if tqdm is not None:
                    tqdm.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} TAG INFO: {log_msg}")
                else:
                    logger.info(log_msg)

        if is_master():
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (i + 1)
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(epoch + 1, time_per_batch, train_loader.batch_size / time_per_batch))

            eval_model = model.module if hasattr(model, "module") else model
            eval_result = test_tse(eval_model, dataloader['test_loader'], 77, config.device)
            rank_1, rank_5, rank_10, map = eval_result['r1'], eval_result['r5'], eval_result['r10'], eval_result['mAP']
            logger.info('Acc@1 {top1:.5f} Acc@5 {top5:.5f} Acc@10 {top10:.5f} mAP {mAP:.5f}'.format(
                top1=rank_1, top5=rank_5, top10=rank_10, mAP=map
            ))
            torch.cuda.empty_cache()
            if best_rank_1 < rank_1:
                best_rank_1 = rank_1
                best_epoch = epoch

                save_obj = {
                    'model': eval_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'it': it,
                }
                torch.save(save_obj, os.path.join(config.model.saved_path, 'checkpoint_best.pth'))

    print(f"best Acc@1: {best_rank_1} at epoch {best_epoch + 1}")


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to yaml config file'
    )
    args = parser.parse_args()

    config = parse_config(args.config)

    Path(config.model.saved_path).mkdir(parents=True, exist_ok=True)

    init_distributed_mode(config)
    set_seed(config)

    run(config)
