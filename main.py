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
from options import get_args
# from eva_clip import create_model_and_transforms
from text_utils.logger import setup_logger
def run(config):
    logger = setup_logger('TAG', distributed_rank=get_rank(), save_dir=config.model.saved_path)
    logger.propagate = False
    logger.info(f'\n{config}')
    if config.experiment.l1_only:
        print("Running STRICT L1 baseline")
    if getattr(config.loss, "use_aerial_correction", False):
        print("Running Single-Adapter Aerial Correction")
    # data
    dataloader = build_pedes_data(config)
    train_loader = dataloader['train_loader']
    num_classes = len(train_loader.dataset.person2text)
    config.num_classes = num_classes

    meters = {
        "loss": AverageMeter(),
        "total_loss": AverageMeter(),
        "ga_loss": AverageMeter(),
        "la_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "distill_loss": AverageMeter(),
        "ice_loss": AverageMeter(),
    }
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
                if getattr(config.loss, "use_aerial_correction", False):
                    warmup_epochs = getattr(config.loss, "warmup_epochs", 3)
                    if epoch < warmup_epochs:
                        loss = ret.get('ga_loss', 0) + ret.get('la_loss', 0) + ret.get('id_loss', 0)
                    else:
                        loss = ret.get('total_loss', 0)
                elif getattr(config.experiment, "l1_only", False):
                    # Baseline L1 loss: L_GA + L_LA + lambda_id * L_id
                    loss = ret.get('itc_loss', 0)
                    if 'id_loss' in ret:
                        loss = loss + ret.get('id_loss', 0)
                else:
                    loss = sum([v for k, v in ret.items() if "loss" in k])

            def _mval(x):
                if torch.is_tensor(x):
                    return x.detach().item()
                return x

            batch_size = batch['image'].shape[0]
            meters['loss'].update(loss.item(), batch_size)
            meters['total_loss'].update(_mval(loss), batch_size)
            meters['ga_loss'].update(_mval(ret.get('ga_loss', 0)), batch_size)
            meters['la_loss'].update(_mval(ret.get('la_loss', 0)), batch_size)
            meters['id_loss'].update(_mval(ret.get('id_loss', 0)), batch_size)
            meters['distill_loss'].update(_mval(ret.get('distill_loss', 0)), batch_size)
            meters['ice_loss'].update(_mval(ret.get('ice_loss', 0)), batch_size)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()
            optimizer.zero_grad()
            it += 1

            if (i + 1) % config.log.print_period == 0:
                logger.info(
                    f"Epoch[{epoch + 1}] Iteration[{i + 1}/{len(train_loader)}], "
                    f"total_loss: {meters['total_loss'].val:.4f}, "
                    f"ga_loss: {meters['ga_loss'].val:.4f}, "
                    f"la_loss: {meters['la_loss'].val:.4f}, "
                    f"id_loss: {meters['id_loss'].val:.4f}, "
                    f"distill_loss: {meters['distill_loss'].val:.4f}, "
                    f"ice_loss: {meters['ice_loss'].val:.4f}, "
                    f"Base Lr: {param_group['lr']:.2e}"
                )

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
