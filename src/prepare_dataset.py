from __future__ import annotations

import ast
from pathlib import Path
import requests

import pandas as pd
from datasets import Dataset, DatasetDict
from hydra.utils import get_original_cwd

from src.config import Config


dontpatronizeme_categories_url = r'https://raw.githubusercontent.com/CRLala/NLPLabs-2024/refs/heads/main/Dont_Patronize_Me_Trainingset/dontpatronizeme_categories.tsv'
dontpatronizeme_pcl_url = r'https://raw.githubusercontent.com/CRLala/NLPLabs-2024/refs/heads/main/Dont_Patronize_Me_Trainingset/dontpatronizeme_pcl.tsv'
dev_semeval_parids_labels_url = r'https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/refs/heads/master/semeval-2022/practice%20splits/dev_semeval_parids-labels.csv'
train_semeval_parids_labels_url = r'https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/refs/heads/master/semeval-2022/practice%20splits/train_semeval_parids-labels.csv'
task4_text_url = r'https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/refs/heads/master/semeval-2022/TEST/task4_test.tsv'

PLC_INTEGER_NUM_CLASSES = 5
DATA_DIR = 'data'

def data_dir() -> Path:
    try:
        root = Path(get_original_cwd()).resolve()
    except:
        root = Path.cwd()
    d = root / DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d

def download_url(url: str) -> Path:
    fname = url.split('/')[-1]
    fpath = data_dir() / fname

    if fpath.exists():
        return fpath

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    fpath.write_bytes(response.content)
    return fpath

def get_dpm_cats_df():
    path = download_url(dontpatronizeme_categories_url)

    return pd.read_csv(
        path,
        sep='\t',
        header=4,
        names=[
            'par_id', 'art_id', 'text',
            'keyword', 'country_code',
            'span_start', 'span_finish',
            'span_text', 'plc_category',
            'number_of_annotators',
        ],
    )


def get_dpm_pcl_df():

    path = download_url(dontpatronizeme_pcl_url)

    return pd.read_csv(
        path,
        sep='\t',
        header=4,
        names=[
            'par_id', 'art_id',
            'keyword', 'country_code',
            'text', 'label',
        ],
    )


def get_task4_test_raw_df():
    path = download_url(task4_text_url)

    return pd.read_csv(
        path,
        sep='\t',
        header=None,
        names=[
            'par_id', 'art_id',
            'keyword', 'country_code',
            'text',
        ],
    )

def get_train_labels_df():

    path = download_url(train_semeval_parids_labels_url)
    return pd.read_csv(path)


def get_dev_labels_df():

    path = download_url(dev_semeval_parids_labels_url)
    return pd.read_csv(path)


def get_train_dev_df() -> pd.DataFrame:

    dataset_df = get_dpm_pcl_df()
    train_labels_df = get_train_labels_df()
    dev_labels_df = get_dev_labels_df()

    type_ml = pd.concat([train_labels_df, dev_labels_df], ignore_index=True)
    type_ml = type_ml[['par_id', 'label']].rename(columns={'label': 'type_multilabel'})
    type_ml['par_id'] = type_ml['par_id'].astype(int)

    df = dataset_df[['par_id', 'text', 'label']].copy()
    df = df.rename(columns={'label': 'severity_int'})
    df['par_id'] = df['par_id'].astype(int)
    df['text'] = df['text'].astype(str, copy=False)
    df['severity_int'] = df['severity_int'].astype(int, copy=False)

    df = df.merge(type_ml, on='par_id', how='left')
    df = df.dropna(subset=['type_multilabel']).reset_index(drop=True)

    df['label'] = (df['severity_int'] >= 2).astype(float)
    df['severity_coral'] = df['severity_int'].map(
        lambda sev_int: [float(sev_int>i) for i in range(0,4)]
    )
    df['type_multilabel'] = df['type_multilabel'].map(
        lambda type_list: [float(v) for v in ast.literal_eval(type_list)]
    )

    return df


def get_train_df() -> pd.DataFrame:
    train_dev_df = get_train_dev_df()

    train_labels_df = get_train_labels_df()
    train_order = train_labels_df['par_id'].astype(int).tolist()

    by_id = train_dev_df.set_index('par_id', drop=False)

    present = [pid for pid in train_order if pid in by_id.index]
    missing = [pid for pid in train_order if pid not in by_id.index]
    # assert not missing, f'train par_id missing in df: {len(missing)}'

    train_df = by_id.loc[present].reset_index(drop=True)
    return train_df


def get_dev_df() -> pd.DataFrame:
    train_dev_df = get_train_dev_df()

    dev_labels_df = get_dev_labels_df()
    dev_order = dev_labels_df['par_id'].astype(int).tolist()

    by_id = train_dev_df.set_index('par_id', drop=False)

    present = [pid for pid in dev_order if pid in by_id.index]
    missing = [pid for pid in dev_order if pid not in by_id.index]
    assert not missing, f'dev par_id missing in df {len(missing)}'

    dev_df = by_id.loc[present].reset_index(drop=True)
    return dev_df


def get_test_df() -> pd.DataFrame:
    df = get_task4_test_raw_df()[['text']].copy()
    df['text'] = df['text'].astype(str, copy=False)
    return df


def prepare_split_text(
        df: pd.DataFrame,
        cfg: Config,
        tokeniser,
        max_length: int,
    ) -> Dataset:

    text_col = 'text'
    ds = Dataset.from_pandas(df, preserve_index=False)

    prep_prompt = cfg.train.prep_prompt
    def add_prompt(batch: dict):
        texts = batch[text_col]
        return {text_col: [prep_prompt + t for t in texts]}

    ds = ds.map(add_prompt, batched=True)
    ds = ds.map(
        lambda batch: tokeniser(
            batch[text_col],
            max_length=max_length,
            truncation=True,
            padding=False,
        ),
        batched=True,
        remove_columns=text_col,
    )

    ds.set_format(type='python')
    return ds


def prepare_dataset(cfg: Config, tokeniser, *, include_test: bool = False) -> DatasetDict:
    train_df = get_train_df().reset_index(drop=True)
    dev_df = get_dev_df().reset_index(drop=True)
    test_df = get_test_df().reset_index(drop=True)

    ds_dict = DatasetDict({
        'train': prepare_split_text(train_df, cfg, tokeniser, max_length=cfg.train.train_max_length),
        'dev': prepare_split_text(dev_df, cfg, tokeniser, max_length=cfg.train.eval_max_length),
        'test': prepare_split_text(test_df, cfg, tokeniser, max_length=cfg.train.eval_max_length)
    })

    return DatasetDict(ds_dict)