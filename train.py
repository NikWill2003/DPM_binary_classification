from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

from src.config import Config
from src.model import PCLClassifier
from src.trainer import PCLTrainer
from src.prepare_dataset import prepare_dataset
from src.utils import (
    seed_all, 
    get_wandb_run, 
    compute_binary_sev_metrics,
    write_preds
    )

cs = ConfigStore.instance()
cs.store(name='schema', node=Config)

def run_fold(
        cfg: Config,
        fold: int,
        fold_train_ds,
        fold_val_ds,
        dev_ds,
        test_ds,
        tokeniser,
        fold_dir: Path,
        hydra_run_dir: Path,
    ) -> dict:

    seed_all(cfg.seed + fold)

    run = get_wandb_run(cfg=cfg, hydra_run_dir=hydra_run_dir, fold=fold)

    try:
        model = PCLClassifier.from_cfg(cfg)

        trainer = PCLTrainer(
            cfg=cfg,
            model=model,
            tokeniser=tokeniser,
            train_dataset=fold_train_ds,
            eval_dataset=fold_val_ds,
            out_dir=fold_dir,
            wandb_run=run,
        )

        train_state = trainer.train()

        dev_logged, dev_es_metric, dev_logits = trainer.evaluate_dataset(dev_ds)
        if run is not None:
            run.log(dev_logged)
            run.summary['fold/dev_metric'] = float(dev_es_metric)

        dev_txt = write_preds('dev.txt', fold_dir, dev_logits)
        test_logits = trainer.predict_dataset(test_ds)
        test_txt = write_preds('test.txt', fold_dir, test_logits)

        fold_state = {
            'fold': fold,
            'best_val_metric': float(train_state['best_val_metric']),
            'best_val_step': int(train_state['best_val_step']),
            'dev_metric': float(dev_es_metric),
            'dev_logits': dev_logits.cpu().numpy(),
            'test_logits': test_logits.cpu().numpy(),
            'dev_txt': str(dev_txt),
            'test_txt': str(test_txt),
        }

        if run is not None:
            run.summary.update({
                    'cv/fold': fold,
                    'cv/best_val_metric': fold_state['best_val_metric'],
                    'cv/best_val_step': fold_state['best_val_step'],
                    'cv/dev_metric': fold_state['dev_metric'],
                    'outputs/dev_txt': fold_state['dev_txt'],
                    'outputs/test_txt': fold_state['test_txt'],
                }
            )

        return fold_state

    finally:
        if run is not None:
            run.finish()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_cv(cfg: Config, hydra_run_dir: Path, out_dir: Path):
    tokeniser = AutoTokenizer.from_pretrained(cfg.model.encoder_name)

    ds = prepare_dataset(cfg, tokeniser)
    train_ds = ds['train']
    dev_ds = ds['dev']
    test_ds = ds['test']

    y = np.asarray(train_ds['severity_int'], dtype=int)
    splitter = StratifiedKFold(
        n_splits=cfg.cv.n_splits,
        shuffle=cfg.cv.shuffle,
        random_state=cfg.seed,
    )

    fold_states: list[dict] = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        fold_dir = out_dir / f'fold_{fold}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_train_ds = train_ds.select(list(tr_idx))
        fold_val_ds = train_ds.select(list(va_idx))

        fold_state = run_fold(
            cfg=cfg,
            fold=fold,
            fold_train_ds=fold_train_ds,
            fold_val_ds=fold_val_ds,
            dev_ds=dev_ds,
            test_ds=test_ds,
            tokeniser=tokeniser,
            fold_dir=fold_dir,
            hydra_run_dir=hydra_run_dir,
        )
        fold_states.append(fold_state)

    if cfg.cv.ensemble:
        ensemble_dir = out_dir / 'ensemble'
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        dev_logits_folds = np.stack([fs['dev_logits'].reshape(-1) for fs in fold_states], axis=0)
        test_logits_folds = np.stack([fs['test_logits'].reshape(-1) for fs in fold_states], axis=0)

        dev_logits_ens = dev_logits_folds.mean(axis=0)
        test_logits_ens = test_logits_folds.mean(axis=0)

        dev_txt = write_preds('dev.txt', ensemble_dir, dev_logits_ens)
        test_txt = write_preds('test.txt', ensemble_dir, test_logits_ens)

        ens_run = get_wandb_run(cfg=cfg, hydra_run_dir=hydra_run_dir, fold=None, ensemble=True)
        try:
            dev_labels = np.asarray(dev_ds['label'], dtype=float).reshape(-1)
            dev_probs = torch.sigmoid(torch.from_numpy(dev_logits_ens.reshape(-1))).numpy()
            ens_metrics = compute_binary_sev_metrics(dev_probs, dev_labels)
            ens_es_metric = ens_metrics[cfg.train.early_stop_metric]

            if ens_run is not None:
                ens_run.log({f'dev_metrics/{k}': float(v) for k, v in ens_metrics.items()})
                ens_run.summary.update({
                        'dev_metric': ens_es_metric,
                        'outputs/dev_txt': str(dev_txt),
                        'outputs/test_txt': str(test_txt),
                    }
                )
        finally:
            if ens_run is not None:
                ens_run.finish()


def run_single(cfg: Config, hydra_run_dir: Path, out_dir: Path):
    tokeniser = AutoTokenizer.from_pretrained(cfg.model.encoder_name)

    ds = prepare_dataset(cfg, tokeniser) 
    train_full = ds['train']
    dev_ds = ds['dev']
    test_ds = ds['test']

    split = train_full.train_test_split(
        test_size=cfg.train.eval_size,
        seed=cfg.seed,
        stratify_by_column='severity_int',
    )
    train_ds = split['train']
    eval_ds = split['test']

    run = get_wandb_run(cfg=cfg, hydra_run_dir=hydra_run_dir, fold=None, ensemble=False)

    try:
        model = PCLClassifier.from_cfg(cfg)

        single_dir = out_dir / 'single'
        single_dir.mkdir(parents=True, exist_ok=True)

        trainer = PCLTrainer(
            cfg=cfg,
            model=model,
            tokeniser=tokeniser,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            out_dir=single_dir,
            wandb_run=run,
        )

        train_state = trainer.train()

        dev_logged, dev_es_metric, dev_logits = trainer.evaluate_dataset(dev_ds)
        test_logits = trainer.predict_dataset(test_ds)

        dev_txt = write_preds('dev.txt', single_dir, dev_logits)
        test_txt = write_preds('test.txt', single_dir, test_logits)

        if run is not None:
            run.log(dev_logged)
            run.summary.update({
                    'single/dev_metric': float(dev_es_metric),
                    'outputs/dev_txt': str(dev_txt),
                    'outputs/test_txt': str(test_txt),
                    'outputs/best_model_path': (train_state['best_model_path'] or ''),
                }
            )
    finally:
        if run is not None:
            run.finish()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: Config) -> None:
    hydra_run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    hydra_run_dir.mkdir(parents=True, exist_ok=True)

    out_dir = hydra_run_dir / 'out'
    out_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config=cfg, f=str(out_dir / 'config.yaml'))
    seed_all(cfg.seed)

    if cfg.cv.enabled:
        run_cv(cfg=cfg, hydra_run_dir=hydra_run_dir, out_dir=out_dir)
    else:
        run_single(cfg=cfg, hydra_run_dir=hydra_run_dir, out_dir=out_dir)


if __name__ == '__main__':
    main()