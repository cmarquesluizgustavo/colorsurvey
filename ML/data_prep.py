import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class ColorDataset(Dataset):
    def __init__(self, rgb_data, labels):
        # rgb_data: (N, 3) normalized to [0, 1]
        # labels: (N,) integer class indices
        self.data = torch.FloatTensor(rgb_data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_and_preprocess_data(csv_path, top_n=329):
    """
    Loads data from CSV, filters for common colors, and prepares train/test splits.
    top_n: Keep only the top N most common colors.
    """
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return None, None, None

    # Filter out rare colors
    print("Filtering data...")
    color_counts = df["colorname"].value_counts()
    common_colors = color_counts.nlargest(top_n).index
    df_filtered = df[df["colorname"].isin(common_colors)].copy()

    print(f"Original samples: {len(df):,}, Filtered samples: {len(df_filtered):,}")
    print(f"Number of classes: {len(common_colors)}")
    print(f"Classes: {common_colors.tolist()}")

    # Prepare features (RGB) and labels
    # RGB is 0-255 in CSV, normalize to 0-1
    X = df_filtered[["r", "g", "b"]].values.astype(np.float32) / 255.0
    if X is None:
        raise ValueError("No data available after filtering.")

    y_text = df_filtered["colorname"].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_text)

    return X, y, le
