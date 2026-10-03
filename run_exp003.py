"""
EXP-003: ViT-Tiny HAM10000
=====================================================
This script orchestrates training, evaluation, and attention
visualization using the original repository code, adding only
recording/saving of per-epoch metrics and experiment outputs.

NO changes to: augmentation, loss, optimizer, learning rate,
batch size, split, or number of epochs.

Architecture: vit_tiny_patch16_224.augreg_in21k_ft_in1k (timm)
Pretraining: ImageNet-21K → ImageNet-1K (AugReg)

NOTE: The pretraining source differs from EXP-001/EXP-002 which
use ImageNet-1K only. This is a documented methodological
difference, not an error.
"""
import sys
import os
import json
import time
import random
from pathlib import Path
from datetime import datetime

# Ensure src is importable
sys.path.insert(0, "src")

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ham10000.utils import load_config
from ham10000.data import build_dataloaders, compute_class_weights
from ham10000.models import build_model
from ham10000.train import set_seed, train_one_epoch, validate

# ── Configuration ────────────────────────────────────────────
CONFIG_PATH = "configs/vit_tiny.yaml"
RESULTS_DIR = Path("results/vit_tiny")
ATTENTION_DIR = RESULTS_DIR / "attention_maps"


def generate_attention_maps(model, val_loader, idx_to_class, device, output_dir, n_each=5):
    """Generate ViT attention visualizations for correct and incorrect predictions.

    Extracts multi-head self-attention from the last transformer block,
    averages across heads, and maps the CLS→patch attention onto the
    14×14 patch grid, then upsamples to 224×224 and overlays on the
    original image.

    This is NOT Grad-CAM. It is a direct attention visualization
    appropriate for Vision Transformers.
    """
    from PIL import Image
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Collect correct and incorrect samples
    val_subset = val_loader.dataset
    dataset = val_subset.dataset
    df = dataset.data
    class_to_idx = dataset.class_to_idx
    image_paths = dataset.image_paths

    correct, incorrect = [], []
    indices = list(val_subset.indices)
    random.seed(42)
    random.shuffle(indices)

    for row_index in indices:
        row = df.iloc[row_index]
        image_id = row['image_id']
        true_idx = class_to_idx[row['dx']]
        path = image_paths.get(image_id)
        if path is None:
            continue

        # Prepare image
        img = Image.open(path).convert('RGB').resize((224, 224))
        rgb_img = np.array(img, dtype=np.float32) / 255.0

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = normalize(rgb_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = int(output.argmax(dim=1).item())

        sample = {'path': path, 'true_idx': true_idx,
                  'pred_idx': pred_idx, 'rgb_img': rgb_img,
                  'input_tensor': input_tensor}

        if pred_idx == true_idx and len(correct) < n_each:
            correct.append(sample)
        elif pred_idx != true_idx and len(incorrect) < n_each:
            incorrect.append(sample)

        if len(correct) >= n_each and len(incorrect) >= n_each:
            break

    print(f"  Found {len(correct)} correct and {len(incorrect)} incorrect samples")

    # Register attention hook on the last transformer block
    attn_weights = {}

    def hook_fn(module, input, output):
        # timm VisionTransformer Attention module stores attn after softmax
        # We hook into the Attention module's forward to capture attention
        attn_weights['last'] = output

    # The last block's attention module
    last_block = model.blocks[-1]
    last_attn = last_block.attn

    # We need to extract attention from the Attention module
    # timm's Attention.forward returns x by default; we need the attention map
    # Use a different approach: register forward hook on the attn_drop layer
    # which receives the attention weights after softmax
    attn_map_storage = {}

    def attn_hook(module, input, output):
        # input[0] is the attention weights after softmax, before dropout
        attn_map_storage['attn'] = input[0].detach().cpu()

    hook = last_attn.attn_drop.register_forward_hook(attn_hook)

    def visualize_sample(sample, prefix, index):
        input_tensor = sample['input_tensor']
        rgb_img = sample['rgb_img']
        true_cls = idx_to_class[sample['true_idx']]
        pred_cls = idx_to_class[sample['pred_idx']]

        attn_map_storage.clear()
        with torch.no_grad():
            _ = model(input_tensor)

        if 'attn' not in attn_map_storage:
            print(f"  Warning: could not capture attention for {prefix}_{index}")
            return

        # attn shape: [batch, heads, tokens, tokens]
        attn = attn_map_storage['attn'][0]  # [heads, tokens, tokens]
        # Average across heads
        attn_avg = attn.mean(dim=0)  # [tokens, tokens]
        # CLS token attention to patch tokens (row 0, columns 1:)
        cls_attn = attn_avg[0, 1:]  # [196]
        # Reshape to 14x14
        cls_attn = cls_attn.reshape(14, 14).numpy()
        # Normalize to [0, 1]
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
        # Upsample to 224x224
        from PIL import Image as PILImage
        attn_resized = np.array(PILImage.fromarray(
            (cls_attn * 255).astype(np.uint8)
        ).resize((224, 224), PILImage.BILINEAR)) / 255.0

        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        ax1.imshow(rgb_img)
        ax1.set_title('Original')
        ax1.axis('off')

        ax2.imshow(cls_attn, cmap='hot', interpolation='nearest')
        ax2.set_title('CLS Attention (14×14)')
        ax2.axis('off')

        # Overlay
        ax3.imshow(rgb_img)
        ax3.imshow(attn_resized, cmap='hot', alpha=0.5)
        ax3.set_title(f'True: {true_cls} | Pred: {pred_cls}')
        ax3.axis('off')

        plt.tight_layout()
        out_path = output_dir / f"attention_{prefix}_{index}.png"
        plt.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")

    for i, sample in enumerate(correct, start=1):
        visualize_sample(sample, "correct", i)

    for i, sample in enumerate(incorrect, start=1):
        visualize_sample(sample, "incorrect", i)

    hook.remove()
    print("  Attention visualization complete.")


def main():
    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ATTENTION_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config(CONFIG_PATH)

    # ── 1. Record environment ────────────────────────────────
    import torchvision
    import sklearn
    import pandas
    import matplotlib
    import timm

    env_info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "timm_version": timm.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6, 0) if torch.cuda.is_available() else None,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pandas.__version__,
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
        "platform": sys.platform,
    }
    print("=" * 60)
    print("EXP-003 ENVIRONMENT")
    print("=" * 60)
    for k, v in env_info.items():
        print(f"  {k}: {v}")

    with open(RESULTS_DIR / "environment.json", "w") as f:
        json.dump(env_info, f, indent=2)

    # ── 2. Build dataloaders ─────────────────────────────────
    print("\n" + "=" * 60)
    print("BUILDING DATALOADERS")
    print("=" * 60)
    train_loader, val_loader, class_to_idx = build_dataloaders(config)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    print(f"  Train samples: {n_train}")
    print(f"  Val samples:   {n_val}")
    print(f"  Classes:       {class_to_idx}")

    # ── 3. Setup model, loss, optimizer ──────────────────────
    print("\n" + "=" * 60)
    print("MODEL SETUP")
    print("=" * 60)
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = build_model(config.num_classes, config.model_name).to(device)
    print(f"  Model: {config.model_name}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Feature dimension:    {model.num_features}")

    if config.use_class_weights:
        class_weights = compute_class_weights(train_loader, config.num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"  Class weights: {class_weights.cpu().tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print(f"  Optimizer: Adam, lr={config.learning_rate}")
    print(f"  Loss: CrossEntropyLoss (weighted={config.use_class_weights})")

    # ── 4. Training loop with history recording ──────────────
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    checkpoint_path = RESULTS_DIR / "best_model.pth"
    history = []
    best_val_loss = float('inf')
    best_epoch = -1
    total_train_start = time.time()

    for epoch in range(config.num_epochs):
        epoch_start = time.time()

        # Use the original train_one_epoch and validate functions
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        epoch_duration = time.time() - epoch_start

        # Also compute training accuracy for the record
        _, train_acc = validate(model, train_loader, criterion, device)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_acc": round(val_acc, 6),
            "epoch_duration_sec": round(epoch_duration, 1),
        }
        history.append(epoch_record)

        print(f'Epoch {epoch+1}/{config.num_epochs} | '
              f'train_loss={train_loss:.4f} | '
              f'val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | '
              f'train_acc={train_acc:.4f} | '
              f'time={epoch_duration:.1f}s')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), str(checkpoint_path))
            # Also save to repo root for evaluate.py compatibility
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'  -> Best model saved (val_loss={val_loss:.4f})')

    total_train_time = time.time() - total_train_start

    print(f"\nTraining complete in {total_train_time:.1f}s")
    print(f"Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")

    # Save training history
    training_summary = {
        "experiment_id": "EXP-003",
        "total_training_time_sec": round(total_train_time, 1),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "history": history,
    }
    with open(RESULTS_DIR / "training_history.json", "w") as f:
        json.dump(training_summary, f, indent=2)

    # ── 5. Evaluation ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION (using best checkpoint)")
    print("=" * 60)

    # Load best checkpoint
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    model.to(device)
    model.eval()

    from ham10000.evaluate import get_predictions, save_confusion_matrix
    from sklearn.metrics import classification_report

    all_preds, all_labels = get_predictions(model, val_loader, device)
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Classification report
    report_text = classification_report(all_labels, all_preds, target_names=class_names)
    print(report_text)

    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # Save classification report
    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"EXP-003 Classification Report\n")
        f.write(f"Checkpoint: best_model.pth (epoch {best_epoch})\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(report_text)

    # Save report as JSON too
    with open(RESULTS_DIR / "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    # Confusion matrices
    save_confusion_matrix(all_labels, all_preds, class_names,
                          str(RESULTS_DIR / "confusion_matrix.png"))
    save_confusion_matrix(all_labels, all_preds, class_names,
                          str(RESULTS_DIR / "confusion_matrix_normalized.png"),
                          normalize="true")

    # ── 6. Attention Visualization ───────────────────────────
    print("\n" + "=" * 60)
    print("ATTENTION VISUALIZATION")
    print("=" * 60)
    print("NOTE: Using CLS-token attention (NOT Grad-CAM).")
    print("ViT does not produce CNN-style feature maps.")

    try:
        generate_attention_maps(model, val_loader, idx_to_class, device, ATTENTION_DIR)
    except Exception as e:
        print(f"Attention visualization error: {e}")
        import traceback
        traceback.print_exc()

    # ── 7. Final summary ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXP-003 COMPLETE")
    print("=" * 60)

    accuracy = report_dict.get("accuracy", 0)
    macro = report_dict.get("macro avg", {})
    weighted = report_dict.get("weighted avg", {})

    final_metrics = {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro.get("precision", 0), 4),
        "macro_recall": round(macro.get("recall", 0), 4),
        "macro_f1": round(macro.get("f1-score", 0), 4),
        "weighted_precision": round(weighted.get("precision", 0), 4),
        "weighted_recall": round(weighted.get("recall", 0), 4),
        "weighted_f1": round(weighted.get("f1-score", 0), 4),
    }

    print(f"\n  Accuracy:         {final_metrics['accuracy']}")
    print(f"  Macro Precision:  {final_metrics['macro_precision']}")
    print(f"  Macro Recall:     {final_metrics['macro_recall']}")
    print(f"  Macro F1:         {final_metrics['macro_f1']}")
    print(f"  Weighted F1:      {final_metrics['weighted_f1']}")
    print(f"  Best Epoch:       {best_epoch}")
    print(f"  Training Time:    {total_train_time:.1f}s")
    print(f"  Checkpoint:       {checkpoint_path}")
    print(f"  Total Params:     {total_params:,}")
    print(f"  Trainable Params: {trainable_params:,}")

    # Per-class
    print(f"\n  Per-class metrics:")
    for cls in class_names:
        c = report_dict.get(cls, {})
        print(f"    {cls:>6s}:  P={c.get('precision',0):.2f}  R={c.get('recall',0):.2f}  F1={c.get('f1-score',0):.2f}  support={c.get('support',0)}")

    # Save final metrics
    with open(RESULTS_DIR / "final_metrics.json", "w") as f:
        json.dump({
            "experiment_id": "EXP-003",
            "final_metrics": final_metrics,
            "per_class": {cls: report_dict[cls] for cls in class_names},
            "best_epoch": best_epoch,
            "total_training_time_sec": round(total_train_time, 1),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "feature_dimension": 192,
            "checkpoint_path": str(checkpoint_path),
            "pretraining_note": "ImageNet-21K -> ImageNet-1K (AugReg), differs from EXP-001/EXP-002 which use ImageNet-1K only",
        }, f, indent=2)

    # Save experiment config
    with open(RESULTS_DIR / "experiment_config.json", "w") as f:
        json.dump({
            "seed": config.seed,
            "model_name": config.model_name,
            "timm_model": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
            "num_classes": config.num_classes,
            "image_size": config.image_size,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
            "optimizer": config.optimizer,
            "use_class_weights": config.use_class_weights,
            "use_augmentation": config.use_augmentation,
            "train_test_split": config.train_test_split,
            "metadata_csv": config.metadata_csv,
            "data_dir": config.data_dir,
            "feature_dimension": 192,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "visualization": "CLS-token attention (not Grad-CAM)",
            "pretraining": "ImageNet-21K -> ImageNet-1K (AugReg)",
        }, f, indent=2)

    print(f"\nAll outputs saved to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
