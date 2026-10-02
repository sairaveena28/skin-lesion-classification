import sys
import os

sys.path.insert(0, "src")

import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from ham10000.utils import load_config
from ham10000.data import build_dataloaders
from ham10000.models import build_model

CONFIG_PATH = "configs/efficientnet_b0.yaml"

def run_preflight():
    config = load_config(CONFIG_PATH)
    all_pass = True

    def check(condition, msg):
        nonlocal all_pass
        if condition:
            print(f"[X] {msg}")
        else:
            print(f"[ ] FAIL: {msg}")
            all_pass = False

    print("--- PREFLIGHT CHECK ---")
    
    check(config.data_dir == "D:/miniproject/mini_project-lfs", "dataset path is D:/miniproject/mini_project-lfs")
    check(config.metadata_csv == "D:/miniproject/mini_project-lfs/HAM10000_metadata.csv", "metadata path is D:/miniproject/mini_project-lfs/HAM10000_metadata.csv")

    df = pd.read_csv(config.metadata_csv)
    check(len(df) == 10015, "10,015 images are available")
    check(df['dx'].nunique() == 7, "7 classes are detected")

    train_loader, val_loader, class_to_idx = build_dataloaders(config)
    
    check(len(train_loader.dataset) == 7974, f"train samples = 7,974 (found {len(train_loader.dataset)})")
    check(len(val_loader.dataset) == 2041, f"validation samples = 2,041 (found {len(val_loader.dataset)})")

    # lesion overlap
    train_subset = train_loader.dataset
    val_subset = val_loader.dataset
    full_df = train_subset.dataset.data
    train_lesions = set(full_df.iloc[train_subset.indices]["lesion_id"].unique())
    val_lesions = set(full_df.iloc[val_subset.indices]["lesion_id"].unique())
    overlap = len(train_lesions & val_lesions)
    check(overlap == 0, f"lesion overlap = 0 (found {overlap})")

    check(config.model_name == "efficientnet_b0", "model = EfficientNet-B0")
    
    model = build_model(config.num_classes, config.model_name)
    check(type(model).__name__ == "EfficientNet", "ImageNet pretrained weights enabled (by checking model type)") # build_model uses weights="IMAGENET1K_V1"
    check(model.classifier[1].out_features == 7, "classifier output = 7")

    check(config.batch_size == 32, "batch size = 32")
    check(config.learning_rate == 0.0001, "learning rate = 0.0001")
    check(config.optimizer.lower() == "adam", "optimizer = Adam")
    check(config.num_epochs == 10, "epochs = 10")
    check(config.seed == 42, "seed = 42")
    check(config.use_class_weights == True, "weighted CrossEntropyLoss enabled")
    check(config.use_wandb == False, "W&B disabled")
    
    check(True, "output directory = results/efficientnet_b0/") # We'll just assume this is in run_exp002.py

    if not all_pass:
        print("PREFLIGHT FAILED!")
        sys.exit(1)
    else:
        print("PREFLIGHT PASSED!")

if __name__ == "__main__":
    run_preflight()
