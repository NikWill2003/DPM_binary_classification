# BestModel

This folder points to the implementation of the best-performing model described in the report. The core code lives in `src/`; this README links to the exact components.

Weights are provided via Google Drive: https://drive.google.com/drive/folders/1md0LZDrtph17YcO5gDjTFDpL4Un3j-HX?usp=sharing. Predictions in `dev.txt` and `test.txt` at the repo root.

[![W&B Run](https://img.shields.io/badge/W%26B-run-blue)](https://wandb.ai/niks_priv/nlp-cw-best-model/table?nw=nwusernikiwillems9)

---

## Main components

- **Model definition (RoBERTa + heads):** 
  https://github.com/NikWill2003/DPM_binary_classification/blob/main/src/model.py#L11-L72

- **Training loop + loss composition:** 
  https://github.com/NikWill2003/DPM_binary_classification/blob/main/src/trainer.py#L28-L419

- **Dataset preparation / split ordering:**
  https://github.com/NikWill2003/DPM_binary_classification/blob/main/src/prepare_dataset.py#L133-L213

- **Metrics + writing predictions:**  
  https://github.com/NikWill2003/DPM_binary_classification/blob/main/src/utils.py#L114-L177

---

## Additions described in the report

- **Encoder upgrade (RoBERTa-large):** 
  https://github.com/NikWill2003/DPM_binary_classification/blob/main/src/model.py#L22-L22

- **CORAL ordinal objective:** 
  label construction: https://github.com/NikWill2003/DPM_binary_classification/blob/main/src/prepare_dataset.py#L103-L130

- **Class-weighted loss:**  
  https://github.com/NikWill2003/DPM_binary_classification/blob/main/src/trainer.py#L171-L185

- **Focal scaling:** 
  https://github.com/NikWill2003/DPM_binary_classification/blob/main/src/trainer.py#L186-L220

- **Auxiliary multi-label PCL-type head:**
  loss definition: https://github.com/NikWill2003/DPM_binary_classification/blob/main/src/trainer.py#L212-L217
  trainer usage: https://github.com/NikWill2003/DPM_binary_classification/blob/main/src/trainer.py#L289-L324

- **LLRD:** parameter grouping functions  
  https://github.com/NikWill2003/DPM_binary_classification/blob/main/src/param_groups.py#L21-L67

- **Cross-fold ensembling:**  
  https://github.com/NikWill2003/DPM_binary_classification/blob/main/train.py#L135-L166

---

## Best run command

```
python train.py \
  seed=0 \
  cv.enabled=true \
  cv.n_splits=4 \
  cv.shuffle=true \
  cv.ensemble=true \
  train.save_best = true \
  train.mixed_precision=bf16 \
  train.max_steps=2000 \
  train.eval_every_steps=100 \
  train.early_stopping_evals=4 \
  optim.optim=adamw \
  optim.wd=0.01 \
  optim.warmup_ratio=0.05 \
  optim.lldr.enabled=true \
  optim.lldr.lr_head=1e-5 \
  optim.lldr.lr_top=1e-5 \
  optim.lldr.layer_decay=0.95 \
  loss.use_coral_objective=true \
  loss.use_weighted_loss=true \
  loss.focal_loss_lambda=1.0 \
  loss.aux_multilabel_plc_loss_weight=0.5 \
  loss.use_weighted_aux_loss=false \
  model.encoder_name=roberta-large \
  model.head_dropout=0.1 \
  model.use_linear_head=true
```