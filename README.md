# DPM_binary_classification - NLP coursework

## Links for marker

- **Required prediction files (repo root):**
  - `dev.txt` — one prediction per line (0/1)
  - `test.txt` — one prediction per line (0/1)
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
