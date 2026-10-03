"""
EXP-003 Preflight: ViT-Tiny HAM10000
======================================
Validates dataset, split, model construction, and CUDA before training.
Does NOT train.
"""
import sys
import os

sys.path.insert(0, "src")

import pandas as pd
from pathlib import Path
import torch
from ham10000.utils import load_config
from ham10000.data import build_dataloaders
from ham10000.models import build_model

CONFIG_PATH = "configs/vit_tiny.yaml"

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

    print("=" * 60)
    print("EXP-003 PREFLIGHT CHECK — ViT-Tiny")
    print("=" * 60)

    # --- Dataset ---
    check(config.data_dir == "D:/miniproject/mini_project-lfs",
          "dataset path is D:/miniproject/mini_project-lfs")
    check(config.metadata_csv == "D:/miniproject/mini_project-lfs/HAM10000_metadata.csv",
          "metadata path is D:/miniproject/mini_project-lfs/HAM10000_metadata.csv")

    df = pd.read_csv(config.metadata_csv)
    check(len(df) == 10015, f"10,015 images are available (found {len(df)})")
    check(df['dx'].nunique() == 7, f"7 classes are detected (found {df['dx'].nunique()})")

    # --- Split ---
    train_loader, val_loader, class_to_idx = build_dataloaders(config)

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    check(n_train == 7974, f"train samples = 7,974 (found {n_train})")
    check(n_val == 2041, f"validation samples = 2,041 (found {n_val})")

    # Lesion overlap
    train_subset = train_loader.dataset
    val_subset = val_loader.dataset
    full_df = train_subset.dataset.data
    train_lesions = set(full_df.iloc[train_subset.indices]["lesion_id"].unique())
    val_lesions = set(full_df.iloc[val_subset.indices]["lesion_id"].unique())
    overlap = len(train_lesions & val_lesions)
    check(overlap == 0, f"lesion overlap = 0 (found {overlap})")

    # --- Config ---
    check(config.model_name == "vit_tiny", f"model = vit_tiny (found {config.model_name})")
    check(config.num_classes == 7, f"num_classes = 7 (found {config.num_classes})")
    check(config.batch_size == 32, f"batch size = 32 (found {config.batch_size})")
    check(config.learning_rate == 0.0001, f"learning rate = 0.0001 (found {config.learning_rate})")
    check(config.optimizer.lower() == "adam", f"optimizer = Adam (found {config.optimizer})")
    check(config.num_epochs == 10, f"epochs = 10 (found {config.num_epochs})")
    check(config.seed == 42, f"seed = 42 (found {config.seed})")
    check(config.use_class_weights == True, "weighted CrossEntropyLoss enabled")
    check(config.use_wandb == False, "W&B disabled")

    # --- Model construction (pretrained=False to avoid unnecessary download) ---
    print("\n--- Model Construction ---")
    import timm
    model = timm.create_model(
        "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        pretrained=False,
        num_classes=7,
    )
    check(model.__class__.__name__ == "VisionTransformer",
          f"model class = VisionTransformer (found {model.__class__.__name__})")
    check(model.num_features == 192, f"feature dimension = 192 (found {model.num_features})")
    check(model.num_classes == 7, f"classifier output = 7 (found {model.num_classes})")

    cfg_input = model.default_cfg.get("input_size", None)
    check(cfg_input == (3, 224, 224), f"default input size = (3, 224, 224) (found {cfg_input})")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    check(total_params == trainable_params, "all parameters are trainable (full fine-tuning)")

    # --- CUDA ---
    print("\n--- CUDA ---")
    cuda_available = torch.cuda.is_available()
    check(cuda_available, "CUDA is available")
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2) if hasattr(torch.cuda.get_device_properties(0), 'total_mem') else round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {gpu_mem} GB")

    # --- Forward-pass smoke test ---
    print("\n--- Forward-Pass Smoke Test ---")
    device = torch.device("cuda" if cuda_available else "cpu")
    model = model.to(device).eval()
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        y = model(x)
    check(tuple(y.shape) == (2, 7), f"output shape = (2, 7) (found {tuple(y.shape)})")
    check(torch.isfinite(y).all().item(), "output is finite")
    print(f"  Device: {device}")
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(y.shape)}")

    # --- Batch size 32 CUDA forward test ---
    if cuda_available:
        print("\n--- Batch-32 CUDA Forward Test ---")
        try:
            x32 = torch.randn(32, 3, 224, 224, device=device)
            with torch.no_grad():
                y32 = model(x32)
            check(tuple(y32.shape) == (32, 7), f"batch-32 output shape = (32, 7) (found {tuple(y32.shape)})")
            check(torch.isfinite(y32).all().item(), "batch-32 output is finite")
            print("  No OOM at batch size 32")
            del x32, y32
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                check(False, f"OOM at batch size 32: {e}")
            else:
                raise

    # --- Feature extraction API ---
    print("\n--- Feature Extraction API ---")
    x_feat = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        features = model.forward_features(x_feat)
    print(f"  forward_features shape: {tuple(features.shape)}")
    print(f"  num_features: {model.num_features}")

    del model, x, x_feat
    if cuda_available:
        torch.cuda.empty_cache()

    # --- EXP-001 / EXP-002 Protection ---
    print("\n--- Experiment Protection ---")
    check(Path("results/baseline_resnet18/best_model.pth").exists(),
          "EXP-001 checkpoint exists")
    check(Path("results/efficientnet_b0/best_model.pth").exists(),
          "EXP-002 checkpoint exists")

    # Verify existing models still build correctly
    resnet = build_model(7, "resnet18")
    check(resnet.fc.out_features == 7, "ResNet18 still builds with 7 classes")
    from torchvision.models import EfficientNet
    effnet = build_model(7, "efficientnet_b0")
    check(effnet.classifier[1].out_features == 7, "EfficientNet-B0 still builds with 7 classes")

    # --- Summary ---
    print("\n" + "=" * 60)
    if all_pass:
        print("EXP-003 PREFLIGHT: PASSED")
    else:
        print("EXP-003 PREFLIGHT: FAILED")
    print("=" * 60)

    if not all_pass:
        sys.exit(1)

if __name__ == "__main__":
    run_preflight()
