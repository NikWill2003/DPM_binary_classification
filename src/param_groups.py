import torch.nn as nn
from typing import Protocol,Optional

from src.model import PCLClassifier

def get_named_grad_params(module: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(n, p) for n, p in module.named_parameters() if p.requires_grad]

def get_named_head_params(model) -> list[tuple[str, nn.Parameter]]:
    head_params = []
    head_params += get_named_grad_params(model.main_head)

    if model.type_multilabel_head is not None:
        head_params += get_named_grad_params(model.type_multilabel_head)

    return head_params

def get_named_encoder_params(model: PCLClassifier) -> list[tuple[str, nn.Parameter]]:
    return get_named_grad_params(model.encoder)

def get_named_roberta_llrd_param_groups(
    roberta_model: nn.Module,
    lr_top: float,
    layer_decay: float,
):
    top_level_params = []
    top_level_params += get_named_grad_params(roberta_model.pooler)  # type: ignore

    param_groups = []
    current_lr = lr_top

    layers = roberta_model.encoder.layer  # type: ignore
    for i in reversed(range(len(layers))): # type: ignore
        lp = get_named_grad_params(layers[i]) # type: ignore
        if lp:
            param_groups.append({'named_params': lp, 'lr': current_lr})
        current_lr *= layer_decay

    emb = get_named_grad_params(roberta_model.embeddings)  # type: ignore
    if emb:
        param_groups.append({'named_params': emb, 'lr': current_lr})

    return top_level_params, param_groups

def get_named_llrd_param_groups(
    model: PCLClassifier, 
    lr_head: float,
    lr_top: float,
    layer_decay: float,
) -> list[dict]:
    
    head_params = get_named_head_params(model)
    
    enc_top_level_params, enc_param_groups = get_named_roberta_llrd_param_groups(
        model.encoder, lr_top, layer_decay
    )

    param_groups = []
    if head_params:
        param_groups.append({'named_params': head_params, 'lr': lr_head})
    
    if enc_top_level_params:
        param_groups.append({'named_params': enc_top_level_params, 'lr': lr_top})

    param_groups.extend(enc_param_groups)

    return param_groups

def is_no_decay(name: str, p: nn.Parameter) -> bool:
    if name.endswith('bias'):
        return True
    if p.ndim == 1: # layernorm weights are 1d
        return True
    return False

def split_named_groups_for_wd(
    named_groups: list[dict],
    wd: float
) -> list[dict]:

    out = []
    for g in named_groups:
        named_params = g['named_params']

        decay_params = []
        no_decay_params = []

        for name, p in named_params:
            if not p.requires_grad:
                continue
            if is_no_decay(name, p):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        lr = g.get('lr', None)

        if decay_params:
            decay_group = {'params': decay_params, 'weight_decay': wd}
            if lr is not None:
                decay_group['lr'] = lr
            out.append(decay_group)
                
        if no_decay_params:
            no_decay_group = {'params': no_decay_params, 'weight_decay': 0.0}
            if lr is not None:
                no_decay_group['lr'] = lr
            out.append(no_decay_group)

    return out
