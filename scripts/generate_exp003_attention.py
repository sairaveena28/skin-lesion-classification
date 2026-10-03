import sys
import os
from pathlib import Path
import random

sys.path.insert(0, "src")

import numpy as np
import torch
import torch.nn as nn
from ham10000.utils import load_config
from ham10000.data import build_dataloaders
from ham10000.models import build_model
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from torchvision import transforms

CONFIG_PATH = "configs/vit_tiny.yaml"
CHECKPOINT_PATH = "results/vit_tiny/best_model.pth"
OUTPUT_DIR = Path("results/vit_tiny/attention_maps")

def generate_attention_maps():
    config = load_config(CONFIG_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading model...")
    model = build_model(config.num_classes, config.model_name)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    # FIX: Disable fused attention on the last block so the attn_drop hook is actually called
    if hasattr(model.blocks[-1].attn, 'fused_attn'):
        model.blocks[-1].attn.fused_attn = False

    print("Loading dataloader...")
    _, val_loader, class_to_idx = build_dataloaders(config)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    val_subset = val_loader.dataset
    dataset = val_subset.dataset
    df = dataset.data
    image_paths = dataset.image_paths
    
    correct, incorrect = [], []
    indices = list(val_subset.indices)
    random.seed(42)
    random.shuffle(indices)
    
    print("Collecting 5 correct and 5 incorrect samples...")
    for row_index in indices:
        row = df.iloc[row_index]
        image_id = row['image_id']
        true_idx = class_to_idx[row['dx']]
        path = image_paths.get(image_id)
        if path is None:
            continue
            
        img = PILImage.open(path).convert('RGB').resize((224, 224))
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
                  'input_tensor': input_tensor, 'index': len(correct) + 1 if pred_idx == true_idx else len(incorrect) + 1}
                  
        if pred_idx == true_idx and len(correct) < 5:
            correct.append(sample)
        elif pred_idx != true_idx and len(incorrect) < 5:
            incorrect.append(sample)
            
        if len(correct) >= 5 and len(incorrect) >= 5:
            break
            
    attn_map_storage = {}
    def attn_hook(module, input, output):
        attn_map_storage['attn'] = input[0].detach().cpu()

    hook = model.blocks[-1].attn.attn_drop.register_forward_hook(attn_hook)
    
    def visualize_sample(sample, prefix):
        input_tensor = sample['input_tensor']
        rgb_img = sample['rgb_img']
        true_cls = idx_to_class[sample['true_idx']]
        pred_cls = idx_to_class[sample['pred_idx']]
        idx = sample['index']
        
        attn_map_storage.clear()
        with torch.no_grad():
            _ = model(input_tensor)
            
        if 'attn' not in attn_map_storage:
            print(f"  Warning: could not capture attention for {prefix}_{idx}")
            return
            
        attn = attn_map_storage['attn'][0]  # [heads, tokens, tokens]
        attn_avg = attn.mean(dim=0)  # [tokens, tokens]
        cls_attn = attn_avg[0, 1:]  # [196]
        
        cls_attn = cls_attn.reshape(14, 14).numpy()
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
        
        attn_resized = np.array(PILImage.fromarray(
            (cls_attn * 255).astype(np.uint8)
        ).resize((224, 224), PILImage.BILINEAR)) / 255.0
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        
        ax1.imshow(rgb_img)
        ax1.set_title('Original')
        ax1.axis('off')
        
        ax2.imshow(cls_attn, cmap='hot', interpolation='nearest')
        ax2.set_title('CLS Attention (14x14)')
        ax2.axis('off')
        
        ax3.imshow(rgb_img)
        ax3.imshow(attn_resized, cmap='hot', alpha=0.5)
        ax3.set_title(f'True: {true_cls} | Pred: {pred_cls}')
        ax3.axis('off')
        
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"{prefix}_{idx:02d}_true_{true_cls}_pred_{pred_cls}.png"
        plt.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")
        
    print("Generating visualizations...")
    for sample in correct:
        visualize_sample(sample, "correct")
    for sample in incorrect:
        visualize_sample(sample, "incorrect")
        
    hook.remove()
    print("Done.")

if __name__ == "__main__":
    generate_attention_maps()
