import os
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import CLASS_NAMES, SEED

def collect_images(root):
    rows = []
    root = os.fspath(root)
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            filepath = os.path.join(dirpath, filename)
            parts = set(os.path.normpath(filepath).split(os.sep))
            label = next((c for c in CLASS_NAMES if c in parts), None)
            if label:
                rows.append({"filepath": filepath, "label": label})
    return pd.DataFrame(rows)

def make_splits(df):
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=SEED
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=SEED
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
