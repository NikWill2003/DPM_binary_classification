# DPM_binary_classification - NLP coursework

## Info for markers

- **Required prediction files:**
  - `dev.txt` — in root
  - `test.txt` — in root
- **Best model code + pointers:** `BestModel/README.md`
  - Contains links to the exact `src/` files/functions implementing each component
  - Contains the **best-run command**
  - Contains the **Google Drive link** for the best models weights (multiple models as we ensemble)

The full command used for the best run is in:
- `BestModel/README.md`

## Repository structure

- `BestModel/` — marker entry point
- `src/` — core implementation (model, trainer, data preparation, LLRD param groups, metrics/utilities)
- `conf/` — Hydra configs
- `train.py` — training entry point
- `environment.yml` — conda env



[![W&B Run](https://img.shields.io/badge/W%26B-run-blue)](https://wandb.ai/niks_priv/nlp-cw-best-model/table?nw=nwusernikiwillems9)