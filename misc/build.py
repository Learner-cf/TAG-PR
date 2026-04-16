import os
import torch
import numpy as np
import math
import torch.nn.functional as F


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb = posemb.unsqueeze(0)
    posemb_new = posemb_new.unsqueeze(0)

    num_new_tokens = posemb_new.shape[1] - hight * width
    num_old_tokens = posemb.shape[1] - int(math.sqrt(posemb.shape[1] - 1)) ** 2
    if num_old_tokens < 1:
        num_old_tokens = 1

    posemb_token = posemb[:, :num_old_tokens]
    posemb_grid = posemb[0, num_old_tokens:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)

    if posemb_token.shape[1] < num_new_tokens:
        extra_tokens = posemb_token[:, :1].repeat(1, num_new_tokens - posemb_token.shape[1], 1)
        posemb_token = torch.cat([posemb_token, extra_tokens], dim=1)
    elif posemb_token.shape[1] > num_new_tokens:
        posemb_token = posemb_token[:, :num_new_tokens]

    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb.squeeze(0)


def interpolate_text(pos_embed_checkpoint, target_dim=77):
    # (n_ctx, n_feat) for pos_embed_checkpoint, including SOT and EOT
    if pos_embed_checkpoint.size(0) == target_dim:
        return pos_embed_checkpoint
    start_token = pos_embed_checkpoint[:1, :]
    end_token = pos_embed_checkpoint[-1:, :]
    pos_tokens = pos_embed_checkpoint[1:-1, :].unsqueeze(0).permute(0, 2, 1)
    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=target_dim - 2, mode='linear')
    pos_tokens = pos_tokens.squeeze(0).t()
    pos_tokens = torch.cat([start_token, pos_tokens, end_token], dim=0)
    return pos_tokens


def load_checkpoint(model, config):
    ckpt = None
    # Load original CLIP checkpoint 
    if config.model.ckpt_type == 'original_clip':
        with open(config.model.checkpoint, 'rb') as opened_file:
            model_tmp = torch.jit.load(opened_file, map_location="cpu")
            state = model_tmp.state_dict()
            
        # Remove unused keys
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in state:
                del state[key]

        # Rename keys and resize tensors 
        new_state = {}
        for name, params in state.items():
            if name == 'visual.positional_embedding' and params.shape != model.visual.positional_embedding.shape:
                params = resize_pos_embed(params, model.visual.positional_embedding, model.visual.num_y, model.visual.num_x)

            if name == 'positional_embedding':
                new_state['encode_text.' + name] = interpolate_text(params, config.experiment.text_length)
            elif name.startswith('transformer') or name in ['positional_embedding', 'token_embedding.weight',
                                                            'ln_final.weight', 'ln_final.bias', 'text_projection']:
                new_state['encode_text.' + name] = params
            else:
                new_state[name] = params
        
    # Load saved checkpoint
    elif config.model.ckpt_type == 'saved':
        ckpt = torch.load(os.path.join(config.model.saved_path, 'checkpoint_best.pth'), map_location='cpu')
        new_state = ckpt['model']
        
    else:
        raise KeyError(f"Unknown ckpt_type: {config.model.ckpt_type}")

    # Load state dict into the model
    load_result = model.load_state_dict(new_state, strict=False)
    
    # Print loading results for verification
    # print(f"Model loaded from {config.model.ckpt_type}. Missing keys: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")

    return model, load_result, ckpt


def cosine_scheduler(config):
    schedule_config = config.schedule
    base_value = schedule_config.lr
    start_warmup_value = schedule_config.lr_start
    final_value = schedule_config.lr_end
    epochs = schedule_config.epoch
    warmup_epochs = schedule_config.epoch_warmup
    niter_per_ep = schedule_config.niter_per_ep

    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def build_optimizer(config, model):
    params = []
    schedule_config = config.schedule
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        weight_decay = schedule_config.weight_decay
        ratio = 1.

        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            weight_decay = 0.
        if "cross" in n or "classifier" in n or "mlm_head" in n:
            ratio = ratio * schedule_config.ratio_factor  # default 5.0
        if "visual_emb_layer" in n:
            ratio = ratio * 10
        if "textual_emb_layer" in n:
            ratio = ratio * 10

        params += [{"params": [p], "weight_decay": weight_decay, "ratio": ratio}]

    optimizer = torch.optim.AdamW(params, lr=schedule_config.lr, betas=schedule_config.betas,
                                  eps=schedule_config.eps, weight_decay=schedule_config.weight_decay)
    return optimizer
