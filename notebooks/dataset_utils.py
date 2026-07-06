"""Shared dataset utilities for NIH ChestX-ray14 multi-label training.

Used by train_densenet121.py, gradcam_exploration.py,
and threshold_optimization.py.

References:
  - Wang et al. (2017): ChestX-ray8, CVPR 2017.
  - Strick et al. (2025): Reproducing and Improving CheXNet.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

import torch
from torch.utils.data import Dataset
from torchvision import transforms


ALL_CONDITIONS: list[str] = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]

DATA_ROOT  = Path("/kaggle/input/datasets/organizations/nih-chest-xrays/data")
CSV_PATH   = DATA_ROOT / "Data_Entry_2017.csv"
IMAGE_DIRS = [DATA_ROOT / f"images_{i:03d}/images" for i in range(1, 13)]
IMAGE_SIZE = 224


def build_image_index() -> dict:
    """Precompute filename→path mapping across all 12 image subdirectories."""
    index = {}
    for d in IMAGE_DIRS:
        if d.exists():
            for p in d.iterdir():
                index[p.name] = p
    return index


def load_multilabel_csv(image_index: dict) -> pd.DataFrame:
    """Load Data_Entry_2017.csv and encode 14 conditions as binary columns."""
    df = pd.read_csv(CSV_PATH, usecols=["Image Index", "Finding Labels", "Patient ID"])
    for cond in ALL_CONDITIONS:
        df[cond] = df["Finding Labels"].apply(
            lambda x: 1 if cond in x.split("|") else 0
        )
    df = df[df["Image Index"].isin(image_index)].reset_index(drop=True)
    return df


def patient_level_split(df: pd.DataFrame):
    """Split into train/val/test at patient level (70/15/15), no patient leakage."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tv_idx, te_idx = next(gss.split(df, groups=df["Patient ID"]))
    df_tv   = df.iloc[tv_idx].reset_index(drop=True)
    df_test = df.iloc[te_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=42)
    tr_idx, va_idx = next(gss2.split(df_tv, groups=df_tv["Patient ID"]))
    df_train = df_tv.iloc[tr_idx].reset_index(drop=True)
    df_val   = df_tv.iloc[va_idx].reset_index(drop=True)

    return df_train, df_val, df_test


class ChestXrayDataset(Dataset):
    """Multi-label chest X-ray dataset returning (image_tensor, label_vector)."""

    def __init__(self, df: pd.DataFrame, image_index: dict, transform):
        self.transform = transform
        self.paths     = [image_index[fn] for fn in df["Image Index"]]
        self.labels    = df[ALL_CONDITIONS].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as img:
            img = img.convert("RGB")
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx])


def get_train_transform():
    """Augmentation pipeline following DannyNet (Strick et al., 2025)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_eval_transform():
    """Deterministic transform for val/test splits."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])