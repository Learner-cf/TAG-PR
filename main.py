import os
import random
import time
from pathlib import Path

import torch
from misc.utils import get_rank
from misc.build import load_checkpoint, cosine_scheduler, build_optimizer
from misc.data import build_pedes_data
from misc.eval import test_tse, test
from misc.utils import parse_config, init_distributed_mode, set_seed, is_master, is_using_distributed, \
    AverageMeter
from model.tbps_model_MoE import clip_vitb
from options import get_args
# from eva_clip import create_model_and_transforms
from text_utils.logger import setup_logger
def run(config):
    logger = setup_logger('TAG', distributed_rank=get_rank())
    logger.propagate = False
    logger.info(f'\n{config}')
    # data
    dataloader = build_pedes_data(config)
    train_loader = dataloader['train_loader']
    num_classes = len(train_loader.dataset.person2text)
    config.num_classes = num_classes

    meters = {
        "loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "itc_tse_loss": AverageMeter(),
        "view_loss": AverageMeter(),
        "view_acc": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "ortho_loss": AverageMeter(),
        "ritc_loss": AverageMeter(),
        "ritc_tse_loss": AverageMeter(),
        "balance_loss": AverageMeter(),
        "router_z_loss": AverageMeter(),
        "id_loss": AverageMeter(),
    }
    best_rank_1 = 0.0
    best_epoch = 0

    # model
    # model_name = "EVA02-CLIP-B-16"
    # pretrained = "/data/zxy/UAV/checkpoints/EVA02_CLIP_B_psz16_s8B.pt"  # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"
    # model, _, preprocess = create_model_and_transforms(config, model_name, pretrained, force_custom_clip=True)
    model = clip_vitb(config, num_classes)
    model.to(config.device)
    model, load_result = load_checkpoint(model, config)

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
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(config.schedule.epoch):
        if is_using_distributed():
            dataloader['train_sampler'].set_epoch(epoch)

        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for i, batch in enumerate(train_loader):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[it] * param_group['ratio']

            if epoch == 0:
                alpha = config.model.softlabel_ratio * min(1.0, i / len(train_loader))
            else:
                alpha = config.model.softlabel_ratio

            with torch.autocast(device_type='cuda'):
                ret = model(batch, alpha, training=True)
                loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['image'].shape[0]
            meters['loss'].update(loss.item(), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['itc_tse_loss'].update(ret.get('itc_tse_loss', 0), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['view_loss'].update(ret.get('view_loss', 0), batch_size)
            meters['ortho_loss'].update(ret.get('ortho_loss', 0), batch_size)
            meters['view_acc'].update(ret.get('view_acc', 0), batch_size)
            meters['ritc_loss'].update(ret.get('ritc_loss', 0), batch_size)
            meters['ritc_tse_loss'].update(ret.get('ritc_tse_loss', 0), batch_size)
            meters['balance_loss'].update(ret.get('balance_loss', 0), batch_size)
            meters['router_z_loss'].update(ret.get('router_z_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()
            optimizer.zero_grad()
            it += 1

            if (i + 1) % config.log.print_period == 0:
                info_str = f"Epoch[{epoch + 1}] Iteration[{i + 1}/{len(train_loader)}]"
                # log loss
                for k, v in meters.items():
                    if v.val != 0:
                        info_str += f", {k}: {v.val:.4f}"
                info_str += f", Base Lr: {param_group['lr']:.2e}"
                logger.info(info_str)

        if is_master():
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (i + 1)
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(epoch + 1, time_per_batch, train_loader.batch_size / time_per_batch))

            eval_result = test_tse(model.module, dataloader['test_loader'], 77, config.device)
            rank_1, rank_5, rank_10, map = eval_result['r1'], eval_result['r5'], eval_result['r10'], eval_result['mAP']
            logger.info('Acc@1 {top1:.5f} Acc@5 {top5:.5f} Acc@10 {top10:.5f} mAP {mAP:.5f}'.format(
                top1=rank_1, top5=rank_5, top10=rank_10, mAP=map
            ))
            torch.cuda.empty_cache()
            if best_rank_1 < rank_1:
                best_rank_1 = rank_1
                best_epoch = epoch

                save_obj = {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                }
                torch.save(save_obj, os.path.join(config.model.saved_path, 'checkpoint_best.pth'))

    print(f"best Acc@1: {best_rank_1} at epoch {best_epoch + 1}")


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == '__main__':

    config_path = 'config/config.yaml'
    config = parse_config(config_path)

    Path(config.model.saved_path).mkdir(parents=True, exist_ok=True)

    init_distributed_mode(config)

    set_seed(config)

    run(config)
