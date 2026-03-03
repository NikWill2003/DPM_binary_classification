import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
    precision_recall_curve,
)


from src.config import Config


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_stamp(hydra_run_dir: Path) -> str:
    hc = HydraConfig.get()
    parts = hydra_run_dir.parts
    is_multirun = str(hc.mode).upper().endswith('MULTIRUN')
    return (
        f'{parts[-3]}/{parts[-2]}/{parts[-1]}' # date/time/job_idx
        if is_multirun
        else f'{parts[-2]}/{parts[-1]}' # date/time
    )


def get_run_name(cfg: Config) -> str:
    enc = str(cfg.model.encoder_name).split('/')[-1]
    parts = [
        enc,
        f'obj-{"coral" if cfg.loss.use_coral_objective else "binary"}',
    ]
    if cfg.loss.aux_multilabel_plc_loss_weight > 0:
        parts.append(f'aux{cfg.loss.aux_multilabel_plc_loss_weight:g}')

    if cfg.optim.lldr is not None:
        lldr = cfg.optim.lldr
        parts.append(f'llrd_h{lldr.lr_head:g}_t{lldr.lr_top:g}_d{lldr.layer_decay:g}')
    else:
        if cfg.optim.optim == 'adamspd':
            parts.append('adam_spd')
    return '_'.join(parts)

def get_wandb_run(
        cfg: Config,
        hydra_run_dir: Path,
        fold: Optional[int],
        ensemble: bool = False
    ):

    if not cfg.wandb.enabled:
        return None

    import wandb

    stamp = get_stamp(hydra_run_dir)
    base_run_name = get_run_name(cfg)

    if fold:
        run_name = f'{base_run_name}-cv-{stamp}-fold-{fold}'
        group = f'{base_run_name}-cv-{stamp}'
        wandb_dir = hydra_run_dir / f'wandb-fold-{fold}'
    elif ensemble:
        run_name = f'{base_run_name}-ensemble-{stamp}'
        group = run_name
        wandb_dir = hydra_run_dir / f'wandb-ensemble'
    else:
        run_name = f'{base_run_name}-{stamp}'
        group = run_name
        wandb_dir = hydra_run_dir / 'wandb'

    wandb_dir.mkdir(parents=True, exist_ok=True)

    return wandb.init(
        project=cfg.wandb.project_name,
        entity=cfg.wandb.entity,
        name=run_name,
        group=group,
        tags=cfg.wandb.tags,
        dir=str(wandb_dir),
        config=OmegaConf.to_container(cfg, resolve=True), # type: ignore
    )

def accum_mean(
        running_mean: dict[str, float],
        running_counts: dict[str, float],
        batch_mean: dict[str, float],
        n: float,
    ) -> tuple[dict[str, float], dict[str, float]]:
        
        if n <= 0:
            return running_mean, running_counts

        for key, value in batch_mean.items():
            if key not in running_mean:
                running_mean[key], running_counts[key] = float(value), float(n)
            else:
                running_mean[key] += (float(value) - running_mean[key]) * float(n) / (running_counts[key] + float(n))
                running_counts[key] += float(n)

        return running_mean, running_counts

def compute_binary_sev_metrics(
    pos_probs: np.ndarray,
    labels: np.ndarray,
):
    labels = labels.reshape(-1).astype(np.int64)

    ap = float(average_precision_score(labels, pos_probs))

    precision, recall, thresholds = precision_recall_curve(labels, pos_probs)
    f1_curve = (2 * precision * recall) / (precision + recall + 1e-12)

    if thresholds.size > 0:
        best_idx = int(np.nanargmax(f1_curve[:-1]))
        best_f1 = float(f1_curve[:-1][best_idx])
        best_thresh = float(thresholds[best_idx])
    else:
        best_f1 = float(np.nanmax(f1_curve))
        best_thresh = 0.5

    out = {
        'ap': ap,
        'best_f1': best_f1,
        'best_threshold': best_thresh,
    }

    preds_05 = (pos_probs >= 0.5).astype(int)
    p05, r05, f05, _ = precision_recall_fscore_support(
        labels, preds_05, average='binary', zero_division=0
    )
    out.update(
        {
            'accuracy@0.5': float(accuracy_score(labels, preds_05)),
            'precision@0.5': float(p05),
            'recall@0.5': float(r05),
            'f1@0.5': float(f05),
        }
    )

    return out

def write_preds(
        file_name: str,
        output_dir: Path,
        logits: np.ndarray | torch.Tensor,
        threshold: float = 0.5,
    ) -> Path:

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / file_name

    if isinstance(logits, torch.Tensor):
        logits_np = logits.detach().cpu().numpy()
    else:
        logits_np = logits

    logits_np = np.asarray(logits_np).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    preds01 = (probs >= threshold).astype(int)

    with out_path.open('w', encoding='utf-8') as f:
        for p in preds01:
            f.write(f'{int(p)}\n')

    return out_path

