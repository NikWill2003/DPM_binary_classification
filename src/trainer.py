from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_scheduler, DataCollatorWithPadding
from datasets import Dataset
from accelerate import Accelerator
from wandb import Run
from tqdm.auto import tqdm

from src.config import Config
from src.param_groups import (
    get_named_llrd_param_groups,
    split_named_groups_for_wd,
    get_named_head_params,
    get_named_encoder_params,
)
from src.model import PCLClassifier
from src.adamSPD import AdamSPD
from src.utils import accum_mean, compute_binary_sev_metrics


class PCLTrainer:

    def __init__(
        self,
        cfg: Config,
        model: PCLClassifier,
        tokeniser,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        out_dir: Path,
        wandb_run: Optional[Run] = None,
    ):
        self.cfg = cfg
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.wandb = wandb_run

        self.main_labels = "severity_coral" if self.cfg.loss.use_coral_objective else "labels"
        self.main_obj = "coral" if self.cfg.loss.use_coral_objective else "binary"

        self.accelerator = Accelerator(
            cpu=(not torch.cuda.is_available()),
            gradient_accumulation_steps=cfg.train.grad_accum,
            mixed_precision=cfg.train.mixed_precision
        )

        self.tokeniser = tokeniser

        self.train_collator = DataCollatorWithPadding(
            self.tokeniser, pad_to_multiple_of=8, padding="longest",
        )
        self.eval_collator = DataCollatorWithPadding(
            self.tokeniser, pad_to_multiple_of=8, padding="longest",
        )

        train_loader = self.get_data_loader(
            train_dataset, self.train_collator, self.cfg.train.train_bs, shuffle=True
        )
        eval_loader = self.get_data_loader(
            eval_dataset, self.eval_collator, self.cfg.train.eval_bs, shuffle=False
        )

        optimiser = self.get_optimiser(model)
        scheduler = self.get_scheduler(optimiser)

        model, optimiser, train_loader, eval_loader, scheduler = self.accelerator.prepare(
            model, optimiser, train_loader, eval_loader, scheduler
        )

        self.main_loss, self.aux_loss = self.get_losses(train_loader)

        self.model = model
        self.optimiser = optimiser
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.scheduler = scheduler

        self.best_value: float = float("-inf") if self.cfg.train.early_stop_higher_is_better else float("inf")
        self.best_step: int = 0
        self.best_state_cpu: Optional[dict[str, torch.Tensor]] = None
        self._is_better = (
            lambda curr, best: curr > best
            if self.cfg.train.early_stop_higher_is_better
            else curr < best
        )
        self.evals_no_improve = 0


    def get_data_loader(self, ds: Dataset, collate_fn, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset=ds, # type: ignore 
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            collate_fn=collate_fn,
        )

    def get_eval_loader(self, dataset: Dataset) -> DataLoader:
        dl = self.get_data_loader(dataset, self.eval_collator, self.cfg.train.eval_bs, shuffle=False)
        return self.accelerator.prepare(dl)

    def get_optimiser(self, model: PCLClassifier) -> torch.optim.Optimizer:
        wd = self.cfg.optim.wd
        llrd = self.cfg.optim.lldr
        lr = self.cfg.optim.lr
        optim_name = self.cfg.optim.optim

        if llrd.enabled:
            named_groups = get_named_llrd_param_groups(
                model,  # type: ignore
                lr_head=float(llrd.lr_head),
                lr_top=float(llrd.lr_top),
                layer_decay=float(llrd.layer_decay),
            )
            param_groups = split_named_groups_for_wd(named_groups, wd=wd)
            return torch.optim.AdamW(param_groups, weight_decay=0.0, fused=True)

        if optim_name == "adamspd":
            device = self.accelerator.device
            spd_lambda = self.cfg.optim.adamspd_lambda

            named_encoder_params = get_named_encoder_params(model)
            encoder_param_groups = split_named_groups_for_wd(
                [{"named_params": named_encoder_params, "lr": lr}],
                wd=spd_lambda,
            )

            spd_param_groups = []
            for group in encoder_param_groups:
                params = group["params"]
                group_lr = group["lr"]
                group_wd = group.get("weight_decay", 0.0)

                pre = None
                if group_wd > 0.0:
                    pre = [p.detach().clone().to(device) for p in params]
                spd_param_groups.append(
                    {"params": params, "lr": group_lr, "weight_decay": group_wd, "pre": pre}
                )

            head_params = [p for _, p in get_named_head_params(model) if p.requires_grad]
            if head_params:
                spd_param_groups.append({"params": head_params, "lr": lr, "weight_decay": 0.0, "pre": None})

            print("using adam spd!")
            return AdamSPD(spd_param_groups, lr=lr)

        param_groups = split_named_groups_for_wd(
            [{"named_params": model.named_parameters(), "lr": lr}],
            wd=wd,
        )
        return torch.optim.AdamW(param_groups, weight_decay=0.0, fused=True)

    def get_scheduler(self, optimiser):
        name = self.cfg.optim.lr_scheduler
        warmup_steps = int(self.cfg.optim.warmup_ratio * self.cfg.train.max_steps)

        return get_scheduler(
            name=name,
            optimizer=optimiser,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.cfg.train.max_steps,
        )

    def get_losses(self, train_loader: DataLoader):
        cfg = self.cfg.loss

        def get_binary_label_weights(col_name: str) -> torch.Tensor:
            pos = None
            n = 0
            for batch in train_loader:
                y = batch[col_name].detach().float()
                pos_b = y.sum(dim=0)
                pos = pos_b if pos is None else (pos + pos_b)
                n += y.shape[0]
            assert pos is not None
            neg = n - pos
            return (neg / (pos + 1e-6)).pow(0.5).clamp(max=10)

        def focal_loss_with_logits(
            logits: torch.Tensor,
            targets: torch.Tensor,
            weight: Optional[torch.Tensor] = None,
            gamma: float = 2.0,
        ):
            targets = targets.float()
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=weight, reduction="none"
            )
            p = torch.sigmoid(logits)
            p_t = p * targets + (1 - p) * (1 - targets)
            loss = (1 - p_t).pow(gamma) * bce
            return loss.mean()

        main_pos_weights = None
        if cfg.use_weighted_loss:
            main_pos_weights = get_binary_label_weights(self.main_labels).to(self.accelerator.device)

        if cfg.focal_loss_lambda > 0:
            main_loss = lambda logits, targets: focal_loss_with_logits(  
                logits, targets, main_pos_weights, cfg.focal_loss_lambda
            )
        else:
            main_loss = nn.BCEWithLogitsLoss(pos_weight=main_pos_weights)

        aux_loss = None
        if cfg.aux_multilabel_plc_loss_weight > 0:
            aux_pos_weight = None
            if cfg.use_weighted_aux_loss:
                aux_pos_weight = get_binary_label_weights("type_multilabel").to(self.accelerator.device)
            aux_loss = nn.BCEWithLogitsLoss(pos_weight=aux_pos_weight)

        return main_loss, aux_loss

    def wandb_log(self, data: dict[str, float | int], step: int) -> None:
        if self.wandb is None or not self.accelerator.is_main_process:
            return
        self.wandb.log(data, step=step)

    def wandb_summary(self, data: dict[str, float | int]) -> None:
        if self.wandb is None or not self.accelerator.is_main_process:
            return
        self.wandb.summary.update(data)
    
    def get_binary_logits(self, logits) -> torch.Tensor:
        if not self.cfg.loss.use_coral_objective:
            return logits.view(-1)
        return logits[:, 1].view(-1)

    def early_stop(self, early_stop_metric: float, opt_step: int) -> bool:
        if self._is_better(early_stop_metric, self.best_value):
            self.best_value = float(early_stop_metric)
            self.best_step = int(opt_step)

            model_unwrapped = self.accelerator.unwrap_model(self.model)
            state = model_unwrapped.state_dict()
            self.best_state_cpu = {k: v.detach().cpu().clone() for k, v in state.items()}

            self.evals_no_improve = 0
        else:
            self.evals_no_improve += 1

        return self.evals_no_improve > self.cfg.train.early_stopping_evals

    @torch.inference_mode()
    def predict_binary(
            self,
            data_loader: DataLoader,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
        
        all_binary_logits = []
        all_binary_labels = []
        for batch in data_loader:
            logits, aux_logits = self.model(batch["input_ids"], batch["attention_mask"])
            all_binary_logits.append(self.get_binary_logits(logits).view(-1))

            labels = batch.get('labels', None)
            if labels is not None:
                all_binary_labels.append(labels.view(-1))

        labels = torch.cat(all_binary_labels) if all_binary_labels else None
        return torch.cat(all_binary_logits), labels

    @torch.inference_mode()
    def predict_dataset(
            self,
            dataset: Dataset,
        ) -> torch.Tensor:

        loader = self.get_eval_loader(dataset)
        preds, _ = self.predict_binary(loader)
        return preds

    @torch.inference_mode()
    def evaluate_dataset(
            self,
            dataset: Dataset,
        ) -> tuple[dict[str, float], float, torch.Tensor]:
        
        loader = self.get_eval_loader(dataset)
        return self.evaluate(loader, mode='dev')

    def step(self, batch, mode: Literal['train', 'eval']) -> tuple[dict[str, float], torch.Tensor]:
        batch_means = {}
        logits, aux_logits = self.model(batch["input_ids"], batch["attention_mask"])

        main_loss = self.main_loss(logits, batch[self.main_labels].view_as(logits))
        total_loss = main_loss

        batch_means[f"{mode}_losses/{self.main_obj}_sev_loss"] = float(main_loss.item())

        if aux_logits is not None and self.aux_loss is not None:
            aux_targets = batch["type_multilabel"]
            aux_loss = self.aux_loss(aux_logits, aux_targets.view_as(aux_logits))
            total_loss = total_loss + aux_loss * self.cfg.loss.aux_multilabel_plc_loss_weight

            batch_means[f"{mode}_losses/aux_ml_loss"] = float(aux_loss.item())

        batch_means[f"{mode}_losses/total_loss"] = float(total_loss.item())

        if mode == 'train':
            self.accelerator.backward(total_loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip_norm)

            self.optimiser.step()
            self.optimiser.zero_grad()
            self.scheduler.step()

            lrs = [pg["lr"] for pg in self.optimiser.param_groups]
            batch_means["train_lr/min"] = float(min(lrs))
            batch_means["train_lr/max"] = float(max(lrs))
        
        binary_logits = self.get_binary_logits(logits)
        return batch_means, binary_logits
    
    @torch.inference_mode()
    def evaluate(
        self, data_loader: DataLoader, mode: Literal['eval', 'dev'] = 'eval'
        ) -> tuple[dict[str, float], float, torch.Tensor]:

        self.model.eval()

        eval_means, counts = {}, {}
        all_binary_logits, all_labels = [], []

        for batch in data_loader:
            batch_means, binary_logits = self.step(batch, mode='eval')

            all_binary_logits.append(binary_logits)
            all_labels.append(batch['labels'].view(-1))

            eval_means, counts = accum_mean(
                eval_means, counts, batch_means, batch['input_ids'].size(0)
                )
            
        all_binary_logits = torch.cat(all_binary_logits).detach()
        labels = torch.cat(all_labels).detach()
        probs = torch.sigmoid(all_binary_logits)

        metrics = compute_binary_sev_metrics(probs.cpu().numpy(), labels.cpu().numpy())
        early_stop_metric = metrics.get(self.cfg.train.early_stop_metric, -1)

        metrics = {f'{mode}_metrics/{key}':val for key, val in metrics.items()}

        return eval_means | metrics, early_stop_metric, all_binary_logits

    def train(self) -> dict:
        cfg = self.cfg.train
        train_means, counts = {}, {}

        batch_step = 0
        max_batch_steps = cfg.max_steps * cfg.grad_accum
        early_stop = False

        pbar = tqdm(total=cfg.max_steps, desc="train", dynamic_ncols=True)

        while (batch_step < max_batch_steps) and (not early_stop):
            for batch in self.train_loader:
                self.model.train()
                opt_step = batch_step // cfg.grad_accum

                with self.accelerator.accumulate(self.model):
                    train_batch_means, _ = self.step(batch, mode='train')

                    if self.accelerator.sync_gradients:
                        pbar.update(1)
                    
                train_means, counts = accum_mean(
                    train_means, counts, train_batch_means, batch["input_ids"].size(0)
                )

                if batch_step % (cfg.log_every_steps * cfg.grad_accum) == 0:
                    self.wandb_log(train_means, opt_step)
                    train_means, counts = {}, {}

                if batch_step % (cfg.eval_every_steps * cfg.grad_accum) == 0:
                    eval_logged, es_metric, _ = self.evaluate(self.eval_loader, mode="eval")
                    self.wandb_log(eval_logged, opt_step)

                    early_stop = self.early_stop(es_metric, opt_step)
                    early_stop = early_stop and (opt_step > 400)

                batch_step += 1 
                if batch_step >= max_batch_steps or early_stop:
                    break

        pbar.close()

        assert self.best_state_cpu is not None
        model_unwrapped = self.accelerator.unwrap_model(self.model)
        model_unwrapped.load_state_dict(self.best_state_cpu, strict=True)
        
        best_path = self.out_dir / "best_model.pt"
        if self.cfg.train.save_best:
            torch.save(self.best_state_cpu, best_path)
            best_model_path = str(best_path)
        else:
            best_model_path = None

        self.wandb_summary(
            {
                f"best_val_{cfg.early_stop_metric}": float(self.best_value),
                "best_val_step": int(self.best_step),
            }
        )

        return {
            "best_model_path": best_model_path,
            "best_val_metric": self.best_value,
            "best_val_step": self.best_step,
        }