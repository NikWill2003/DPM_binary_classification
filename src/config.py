from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class TrainingConfig:
    train_max_length: int = 192
    eval_max_length: int = 256
    max_steps: int = 2000
    train_bs: int = 32

    mixed_precision: str = 'bf16'
    grad_accum: int = 1
    grad_clip_norm: float = 1.0

    eval_bs: int = 256
    log_every_steps: int = 50
    eval_every_steps: int = 100
    eval_size: float = 1/4 # ignored is using cross validation

    early_stop_metric: str = 'ap'
    early_stop_higher_is_better: bool = True
    early_stopping_evals: int = 4

    prep_prompt: str = 'detect patronizing or condescending language: '

    save_best: bool = False

@dataclass
class LLDRConfig:
    enabled: bool = False
    lr_head: float = 5e-5
    lr_top: float = 1e-5
    layer_decay: float = 0.9

@dataclass
class OptimConfig:
    optim: str = 'adamw' # adamw or adamspd
    adamspd_lambda: float = 0.0 # ignored if using adamw

    lr: float = 1e-5
    wd: float = 0.01

    lldr: LLDRConfig = field(default_factory=LLDRConfig)

    lr_scheduler: str = 'linear'
    warmup_ratio: float = 0.05
 
@dataclass
class LossConfig:
    use_coral_objective: bool = False # binary or coral 
    use_weighted_loss: bool = False # just for main loss
    focal_loss_lambda: float = 0.0 # set to 0 to not use focusing term

    aux_multilabel_plc_loss_weight: float = 0.0 # set to 0 to disable the aux term
    use_weighted_aux_loss: bool = False # just for aux loss
    

@dataclass
class ModelConfig:
    encoder_name: str = 'roberta-large'
    head_dropout: float = 0.1
    use_linear_head: bool = True # else multi-layer

@dataclass
class WandBConfig:
    enabled: bool = True
    project_name: str = 'nlp-cw'
    entity: Optional[str] = 'niks_priv'
    tags: tuple[str,...] = ('test',)

@dataclass
class CVConfig:
    enabled: bool = True
    n_splits: int = 4
    shuffle: bool = True
    ensemble: bool = True

@dataclass
class Config:
    seed: int = 0
    verbose: bool = False
    
    cv: CVConfig = field(default_factory=CVConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
