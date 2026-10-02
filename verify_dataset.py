"""
Dataset Integrity & Loader Verification Script
================================================
Verifies that the HAM10000 dataset at the configured path is complete,
correct, and ready for EXP-001 training.

This script does NOT train any model.
"""
import sys
import os

# Ensure src is importable (same as run_exp001.py)
sys.path.insert(0, "src")

import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

from ham10000.utils import load_config
from ham10000.data import Ham10000Dataset, build_dataloaders

CONFIG_PATH = "configs/baseline.yaml"

EXPECTED = {
    "metadata_rows": 10015,
    "images_found": 10015,
    "unique_image_ids": 10015,
    "missing_images": 0,
    "duplicate_image_ids": 0,
    "unique_lesion_ids": 7470,
    "null_image_id": 0,
    "null_lesion_id": 0,
    "null_dx": 0,
    "class_counts": {
        "akiec": 327,
        "bcc": 514,
        "bkl": 1099,
        "df": 115,
        "mel": 1113,
        "nv": 6705,
        "vasc": 142,
    },
    "class_mapping": {
        "akiec": 0,
        "bcc": 1,
        "bkl": 2,
        "df": 3,
        "mel": 4,
        "nv": 5,
        "vasc": 6,
    },
    "train_samples": 7974,
    "val_samples": 2041,
    "unique_train_lesions": 5976,
    "unique_val_lesions": 1494,
    "lesion_overlap": 0,
}


def verify():
    all_pass = True
    results = {}

    def check(name, actual, expected):
        nonlocal all_pass
        passed = actual == expected
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        results[name] = {"actual": actual, "expected": expected, "status": status}
        print(f"  [{status}] {name}: {actual} (expected {expected})")
        return passed

    config = load_config(CONFIG_PATH)

    print("=" * 70)
    print("DATASET INTEGRITY & LOADER VERIFICATION")
    print("=" * 70)
    print(f"\n  Config file:    {CONFIG_PATH}")
    print(f"  metadata_csv:   {config.metadata_csv}")
    print(f"  data_dir:       {config.data_dir}")
    print(f"  seed:           {config.seed}")
    print(f"  train_split:    {config.train_test_split}")

    # ── 1. Metadata CSV checks ──────────────────────────────────
    print(f"\n{'─' * 50}")
    print("1. METADATA CSV CHECKS")
    print(f"{'─' * 50}")

    csv_path = Path(config.metadata_csv)
    print(f"  CSV path: {csv_path.resolve()}")
    assert csv_path.exists(), f"CSV not found: {csv_path}"

    df = pd.read_csv(csv_path)
    check("metadata_rows", len(df), EXPECTED["metadata_rows"])
    check("unique_image_ids", df["image_id"].nunique(), EXPECTED["unique_image_ids"])
    check("duplicate_image_ids", len(df) - df["image_id"].nunique(), EXPECTED["duplicate_image_ids"])
    check("unique_lesion_ids", df["lesion_id"].nunique(), EXPECTED["unique_lesion_ids"])
    check("null_image_id", df["image_id"].isnull().sum(), EXPECTED["null_image_id"])
    check("null_lesion_id", df["lesion_id"].isnull().sum(), EXPECTED["null_lesion_id"])
    check("null_dx", df["dx"].isnull().sum(), EXPECTED["null_dx"])

    # ── 2. Class counts ─────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print("2. CLASS DISTRIBUTION")
    print(f"{'─' * 50}")

    class_counts = df["dx"].value_counts().to_dict()
    for cls_name, expected_count in EXPECTED["class_counts"].items():
        actual_count = class_counts.get(cls_name, 0)
        check(f"class_count_{cls_name}", actual_count, expected_count)

    # ── 3. Image discovery via Dataset class ────────────────────
    print(f"\n{'─' * 50}")
    print("3. IMAGE DISCOVERY (using Ham10000Dataset)")
    print(f"{'─' * 50}")

    # Build dataset with no transform (just for discovery check)
    ds = Ham10000Dataset(config.metadata_csv, config.data_dir, transform=None)

    check("images_found", len(ds.image_paths), EXPECTED["images_found"])

    # Check missing images
    missing = []
    for _, row in df.iterrows():
        if row["image_id"] not in ds.image_paths:
            missing.append(row["image_id"])
    check("missing_images", len(missing), EXPECTED["missing_images"])
    if missing:
        print(f"    First 10 missing: {missing[:10]}")

    # ── 4. Class mapping ────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print("4. CLASS MAPPING")
    print(f"{'─' * 50}")

    print(f"  class_to_idx: {ds.class_to_idx}")
    check("class_mapping", ds.class_to_idx, EXPECTED["class_mapping"])

    # ── 5. Image decode verification ────────────────────────────
    print(f"\n{'─' * 50}")
    print("5. JPEG DECODE VERIFICATION")
    print(f"{'─' * 50}")

    # Test 7 images (one from each class folder)
    test_images = []
    class_folders = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]
    data_dir = Path(config.data_dir)
    for folder in class_folders:
        folder_path = data_dir / folder
        if folder_path.exists():
            jpgs = list(folder_path.glob("*.jpg"))
            if jpgs:
                test_images.append(jpgs[0])

    decode_pass = True
    for img_path in test_images:
        try:
            img = Image.open(img_path)
            img.verify()  # Verify it's a valid image
            # Re-open after verify (verify can't be used after load)
            img = Image.open(img_path)
            img_loaded = img.convert("RGB")
            w, h = img_loaded.size
            file_size = os.path.getsize(img_path)
            print(f"  [PASS] {img_path.parent.name}/{img_path.name}: "
                  f"{w}x{h}, {file_size:,} bytes, format={img.format}")
        except Exception as e:
            decode_pass = False
            all_pass = False
            print(f"  [FAIL] {img_path.name}: {e}")

    if not decode_pass:
        results["jpeg_decode"] = {"status": "FAIL"}
    else:
        results["jpeg_decode"] = {"status": "PASS", "images_tested": len(test_images)}
        print(f"  All {len(test_images)} test images decoded successfully as real JPEGs.")

    # ── 6. Dataloader / split verification ──────────────────────
    print(f"\n{'─' * 50}")
    print("6. DATALOADER & LESION-LEVEL SPLIT")
    print(f"{'─' * 50}")

    train_loader, val_loader, class_to_idx = build_dataloaders(config)

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    check("train_samples", n_train, EXPECTED["train_samples"])
    check("val_samples", n_val, EXPECTED["val_samples"])

    # Verify lesion-level split integrity
    train_subset = train_loader.dataset  # Subset
    val_subset = val_loader.dataset      # Subset

    full_df = train_subset.dataset.data  # underlying Ham10000Dataset.data

    train_lesions = set(full_df.iloc[train_subset.indices]["lesion_id"].unique())
    val_lesions = set(full_df.iloc[val_subset.indices]["lesion_id"].unique())

    check("unique_train_lesions", len(train_lesions), EXPECTED["unique_train_lesions"])
    check("unique_val_lesions", len(val_lesions), EXPECTED["unique_val_lesions"])

    overlap = train_lesions & val_lesions
    check("lesion_overlap", len(overlap), EXPECTED["lesion_overlap"])

    # ── 7. Verify a batch loads correctly ───────────────────────
    print(f"\n{'─' * 50}")
    print("7. BATCH LOAD TEST")
    print(f"{'─' * 50}")

    try:
        batch_imgs, batch_labels = next(iter(train_loader))
        print(f"  [PASS] Train batch: shape={batch_imgs.shape}, "
              f"labels={batch_labels.tolist()}")
        batch_imgs_v, batch_labels_v = next(iter(val_loader))
        print(f"  [PASS] Val batch:   shape={batch_imgs_v.shape}, "
              f"labels={batch_labels_v.tolist()}")
        results["batch_load"] = {"status": "PASS"}
    except Exception as e:
        print(f"  [FAIL] Batch load error: {e}")
        results["batch_load"] = {"status": "FAIL", "error": str(e)}
        all_pass = False

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    if all_pass:
        print("RESULT: ALL CHECKS PASSED ✓")
        print("Dataset is READY for EXP-001 training.")
    else:
        failures = [k for k, v in results.items() if v.get("status") == "FAIL"]
        print(f"RESULT: {len(failures)} CHECK(S) FAILED ✗")
        for f in failures:
            print(f"  FAILED: {f}")
    print(f"{'=' * 70}")

    print("\n*** NO TRAINING WAS STARTED ***")

    return all_pass


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
