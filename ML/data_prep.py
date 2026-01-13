import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


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


def balance_classes(X, y, strategy="undersample", sampling_ratio=1.0, 
                    hybrid_oversample_fraction=0.6, fixed_samples_per_class=None, 
                    random_state=42):
    """
    Balance class distribution using various strategies.
    
    Args:
        X: Feature array (N, 3) with RGB values
        y: Label array (N,) with class indices
        strategy: Balancing strategy to use. Options:
            - "none": No balancing (original distribution)
            - "undersample": Reduce majority classes
            - "oversample_duplicate": Duplicate minority classes
            - "hybrid": Combination of oversampling and undersampling
            - "balanced_fixed": Fix all classes to exact number (requires fixed_samples_per_class)
        sampling_ratio: Target ratio for balancing (0.0 to 1.0)
            - 1.0: Perfect balance (all classes equal to smallest/largest depending on strategy)
            - 0.5: Halfway between current and perfect balance
            - 0.0: No change (same as "none")
        hybrid_oversample_fraction: For hybrid strategy, fraction of balancing done by oversampling (0.0 to 1.0)
            - 0.0: Only undersampling
            - 0.5: Equal mix of over and undersampling
            - 1.0: Only oversampling
        fixed_samples_per_class: For balanced_fixed strategy, target number of samples per class
        random_state: Random seed for reproducibility
    
    Returns:
        X_balanced, y_balanced: Balanced feature and label arrays
    """
    if strategy == "none" or sampling_ratio == 0.0:
        print("No class balancing applied.")
        return X, y
    
    # Print original distribution
    original_counts = Counter(y)
    print(f"\n📊 Original class distribution:")
    print(f"   Min: {min(original_counts.values()):,}, Max: {max(original_counts.values()):,}")
    print(f"   Total samples: {len(y):,}")
    
    # Calculate target distribution
    min_samples = min(original_counts.values())
    max_samples = max(original_counts.values())
    
    if strategy == "undersample":
        # Undersample to smallest class size (or ratio thereof)
        target_size = int(min_samples + (max_samples - min_samples) * (1 - sampling_ratio))
        sampling_strategy = {cls: min(count, target_size) 
                           for cls, count in original_counts.items()}
        
        print(f"⚖️  Applying undersampling (target per class: {target_size:,})...")
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        X_balanced, y_balanced = sampler.fit_resample(X, y)
    
    elif strategy == "oversample_duplicate":
        # Oversample minority classes by duplicating samples
        target_size = int(min_samples + (max_samples - min_samples) * sampling_ratio)
        sampling_strategy = {cls: max(count, target_size) 
                           for cls, count in original_counts.items()}
        
        print(f"⚖️  Applying random oversampling (target per class: {target_size:,})...")
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        X_balanced, y_balanced = sampler.fit_resample(X, y)
    
    elif strategy == "hybrid":
        # Hybrid: Combination of oversampling (duplication) and undersampling
        # This balances the dataset from both ends
        print(f"🔄 Applying hybrid sampling (Oversampling + Undersampling)...")
        print(f"   Mix: {hybrid_oversample_fraction*100:.0f}% oversampling, {(1-hybrid_oversample_fraction)*100:.0f}% undersampling")
        
        # Step 1: Oversample minority classes to intermediate level
        # The oversample fraction determines how much of the balancing is done by oversampling
        intermediate_size = int(min_samples + (max_samples - min_samples) * sampling_ratio * hybrid_oversample_fraction)
        over_strategy = {cls: max(count, intermediate_size) 
                        for cls, count in original_counts.items() if count < intermediate_size}
        
        if over_strategy:  # Only oversample if there are classes to oversample
            over_sampler = RandomOverSampler(sampling_strategy=over_strategy, random_state=random_state)
            X_temp, y_temp = over_sampler.fit_resample(X, y)
        else:
            X_temp, y_temp = X, y
        
        # Step 2: Undersample majority classes to target
        # The remaining balancing is done by undersampling
        target_size = int(min_samples + (max_samples - min_samples) * sampling_ratio)
        under_counts = Counter(y_temp)
        under_strategy = {cls: min(count, target_size) 
                         for cls, count in under_counts.items()}
        
        under_sampler = RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_state)
        X_balanced, y_balanced = under_sampler.fit_resample(X_temp, y_temp)
    
    elif strategy == "balanced_fixed":
        # Fixed balanced: Set all classes to exact same number of samples
        if fixed_samples_per_class is None:
            raise ValueError("balanced_fixed strategy requires fixed_samples_per_class parameter")
        
        target_size = int(fixed_samples_per_class)
        print(f"⚖️  Applying balanced_fixed strategy (target per class: {target_size:,})...")
        
        # Create sampling strategy for all classes
        sampling_strategy = {}
        
        for cls, count in original_counts.items():
            sampling_strategy[cls] = target_size
        
        # First oversample classes below target
        over_strategy = {cls: target_size for cls, count in original_counts.items() if count < target_size}
        
        if over_strategy:
            print(f"   Oversampling {len(over_strategy)} classes below target...")
            over_sampler = RandomOverSampler(sampling_strategy=over_strategy, random_state=random_state)
            X_temp, y_temp = over_sampler.fit_resample(X, y)
        else:
            X_temp, y_temp = X, y
        
        # Then undersample classes above target
        temp_counts = Counter(y_temp)
        under_strategy = {cls: target_size for cls, count in temp_counts.items() if count > target_size}
        
        if under_strategy:
            print(f"   Undersampling {len(under_strategy)} classes above target...")
            under_sampler = RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_state)
            X_balanced, y_balanced = under_sampler.fit_resample(X_temp, y_temp)
        else:
            X_balanced, y_balanced = X_temp, y_temp
    
    else:
        raise ValueError(f"Unknown balancing strategy: {strategy}")
    
    # Print final distribution
    final_counts = Counter(y_balanced)
    print(f"✅ Balanced class distribution:")
    print(f"   Min: {min(final_counts.values()):,}, Max: {max(final_counts.values()):,}")
    print(f"   Total samples: {len(y_balanced):,} (change: {len(y_balanced) - len(y):+,})")
    print(f"   Imbalance ratio: {max(final_counts.values()) / min(final_counts.values()):.2f}:1")
    
    return X_balanced, y_balanced


def load_and_preprocess_data(csv_path, top_n=100, test_size=0.2, 
                             balance_strategy="none", balance_ratio=1.0, 
                             fixed_samples_per_class=None, random_state=42):
    """
    Load data from CSV, filter for common colors, split into train/test, 
    and balance ONLY the training set.
    
    Args:
        csv_path: Path to the CSV file with color data
        top_n: Keep only the top N most common colors
        test_size: Proportion of data to use for testing (0.0 to 1.0)
        balance_strategy: Class balancing strategy (see balance_classes for options)
        balance_ratio: Target balancing ratio (0.0 to 1.0, where 1.0 is perfect balance)
        fixed_samples_per_class: For balanced_fixed strategy, target number of samples per class
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test, le: Split features, labels, and label encoder
    """
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return None, None, None, None, None

    # Filter out rare colors
    print("Filtering data...")
    color_counts = df["colorname"].value_counts()
    common_colors = color_counts.nlargest(top_n).index
    df_filtered = df[df["colorname"].isin(common_colors)].copy()

    print(f"Original samples: {len(df):,}, Filtered samples: {len(df_filtered):,}")
    print(f"Number of classes: {len(common_colors)}")

    # Prepare features (RGB) and labels
    # RGB is 0-255 in CSV, normalize to 0-1
    X = df_filtered[["r", "g", "b"]].values.astype(np.float32) / 255.0
    if X is None:
        raise ValueError("No data available after filtering.")

    y_text = df_filtered["colorname"].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    
    # IMPORTANT: Split into train/test BEFORE balancing
    # This ensures test set maintains original distribution for fair comparison
    print(f"\nSplitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")
    
    # Apply class balancing ONLY to training set
    # Test set remains unbalanced to represent real-world distribution
    X_train_balanced, y_train_balanced = balance_classes(
        X_train, y_train, 
        strategy=balance_strategy, 
        sampling_ratio=balance_ratio,
        fixed_samples_per_class=fixed_samples_per_class,
        random_state=random_state
    )

    return X_train_balanced, X_test, y_train_balanced, y_test, le


class DataLoader:
    """Unified data loader for all training methods."""
    
    def __init__(self, config):
        self.config = config
        self.data_config = config["data"]
        self.seed = config["seed"]
    
    def load(self):
        """Load and prepare data for training."""
        # Load and preprocess data (now includes train/test split and balancing)
        X_train, X_test, y_train, y_test, le = load_and_preprocess_data(
            self.data_config["csv_path"],
            top_n=self.data_config["top_n_colors"],
            test_size=self.data_config["test_size"],
            balance_strategy=self.data_config.get("balance_strategy", "none"),
            balance_ratio=self.data_config.get("balance_ratio", 1.0),
            fixed_samples_per_class=self.data_config.get("fixed_samples_per_class"),
            random_state=self.seed
        )
        
        if X_train is None or y_train is None:
            raise ValueError("Failed to load data")
        
        num_classes = len(le.classes_)
        
        # Prepare data bundle
        data_bundle = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "num_classes": num_classes,
            "label_encoder": le
        }
        
        # Add PyTorch DataLoaders if batch_size is specified (for metric learning)
        if "batch_size" in self.data_config:
            train_dataset = ColorDataset(X_train, y_train)
            test_dataset = ColorDataset(X_test, y_test)
            
            data_bundle["train_loader"] = TorchDataLoader(
                train_dataset,
                batch_size=self.data_config["batch_size"],
                shuffle=True,
                num_workers=6,
                persistent_workers=True
            )
            data_bundle["test_loader"] = TorchDataLoader(
                test_dataset,
                batch_size=self.data_config["batch_size"],
                shuffle=False,
                num_workers=4,
                persistent_workers=True
            )
        
        return data_bundle
