# EXP-003 Research Report

## 1. Experiment Overview
EXP-003 is the third controlled experiment within the "Resource-Efficient and Noise-Aware Hybrid Quantum-Classical Learning for Multiclass Skin Lesion Classification" research project. This experiment implements a classical Vision Transformer (ViT-Tiny/16) baseline on the HAM10000 dataset. It establishes a benchmark for transformer-based self-attention representations, following the prior evaluations of ResNet18 (EXP-001) and EfficientNet-B0 (EXP-002) Convolutional Neural Network (CNN) architectures. 

## 2. Research Objective
The primary objective of EXP-003 is to train, evaluate, and extract a 192-dimensional structural representation using a ViT-Tiny/16 model. By enforcing an identical training protocol and dataset split to the preceding experiments, this baseline isolates architectural performance. The resulting trained feature extractor provides the self-attention pathway for subsequent dimensionality reduction (PCA) and hybrid classical-quantum fusion architectures.

## 3. Dataset

### 3.1 HAM10000 Dataset
The dataset utilized is the HAM10000 benchmark, comprising 10,015 dermatoscopic images of skin lesions spanning seven diagnostic categories.

### 3.2 Class Distribution
The 10,015 images are distributed across seven highly imbalanced classes:
- Actinic keratoses and intraepithelial carcinoma (AKIEC): 327
- Basal cell carcinoma (BCC): 514
- Benign keratosis-like lesions (BKL): 1,099
- Dermatofibroma (DF): 115
- Melanoma (MEL): 1,113
- Melanocytic nevi (NV): 6,705
- Vascular lesions (VASC): 142

### 3.3 Lesion-Level Data Splitting
The data contains 7,470 unique physical lesions. A strict lesion-level train/validation split was instituted utilizing a fixed initialization seed (42).
- Training set: 7,974 images (5,976 unique lesions)
- Validation set: 2,041 images (1,494 unique lesions)

### 3.4 Data Leakage Prevention
The lesion-level split ensured a 0% lesion overlap between the training and validation sets. This methodology strictly prevents model evaluation on alternate images of lesions previously observed during training, eliminating data leakage.

## 4. Model Architecture

### 4.1 ViT-Tiny
The experiment employed the `vit_tiny_patch16_224.augreg_in21k_ft_in1k` architecture from the `timm` library.

### 4.2 Patch Embedding
Images of resolution 224×224 were divided into non-overlapping 16×16 patches, yielding a 14×14 grid, flattening to 196 discrete spatial tokens, plus 1 global CLS token (197 total tokens).

### 4.3 Transformer Encoder
The model features 12 transformer encoder blocks, utilizing 3 parallel self-attention heads per block. 

### 4.4 Classification Head
A `Linear(192, 7)` classification layer replaced the pretrained head to directly output probabilities for the 7 target classes.

### 4.5 Parameter and Feature Dimensions
The configuration extracts a 192-dimensional latent feature vector (via the CLS token). The fully fine-tuned model comprises exactly 5,525,767 total and trainable parameters.

## 5. Pretraining
The ViT-Tiny model leveraged transfer learning initialized with ImageNet-21K pretraining, followed by fine-tuning on ImageNet-1K (`augreg_in21k_ft_in1k`). This constitutes a broader pretraining regime than the strictly ImageNet-1K pretraining utilized in the EXP-001 and EXP-002 baselines.

## 6. Data Preprocessing and Augmentation
Validation inputs were resized to 224×224 pixels and normalized using ImageNet parameters. Training augmentation included:
- RandomResizedCrop(224, scale=(0.8,1.0))
- RandomHorizontalFlip(p=0.5)
- RandomRotation(20)
- ColorJitter(0.1, 0.1, 0.1)

## 7. Training Configuration
The model was subjected to full fine-tuning. Optimization was governed by the Adam optimizer with a fixed learning rate of 1e-4 and a batch size of 32. Training endured for exactly 10 epochs. Complex routines including learning rate scheduling, Test-Time Augmentation (TTA), MixUp, and CutMix were intentionally excluded to preserve a clean baseline metric.

## 8. Class-Imbalance Handling
To counteract the overwhelming prevalence of the NV class, the optimization target was a Weighted Cross-Entropy Loss. Class weights were computed dynamically and exclusively derived from the training set distribution. Oversampling, undersampling, and WeightedRandomSampler mechanisms were prohibited.

## 9. Checkpoint Selection
The final predictive model was chosen purely by identifying the checkpoint that achieved the lowest validation loss across the 10 epochs. 

## 10. Evaluation Metrics
Performance was measured using Accuracy, alongside Macro and Weighted variants of Precision, Recall, and F1-score to correctly interpret performance across the imbalanced distribution. A detailed per-class diagnostic classification report was generated.

## 11. Experimental Results

### 11.1 Overall Metrics
The optimum validation loss (0.550898) was obtained at Epoch 8.
- **Accuracy**: 0.8055
- **Macro Precision**: 0.7084
- **Macro Recall**: 0.7455
- **Macro F1**: 0.7231
- **Weighted Precision**: 0.8230
- **Weighted Recall**: 0.8055
- **Weighted F1**: 0.8114

### 11.2 Per-Class Performance
Per-class F1-scores were reported as follows:
- AKIEC: 0.57
- BCC: 0.78
- BKL: 0.68
- DF: 0.68
- MEL: 0.58
- NV: 0.89
- VASC: 0.88

### 11.3 Confusion Matrix Analysis
The confusion matrices indicate that despite class weighting, precision for severe minority classes (AKIEC and MEL) was hindered by misclassifications originating heavily from the majority classes (NV and BKL). 

## 12. Attention Visualization

### 12.1 Attention Extraction Method
A post-training visualization fix was implemented using the frozen model. Scaled Dot Product Attention was forced out of its fused optimization path strictly during inference to retrieve true attention probabilities from the `attn_drop` layer. 

### 12.2 CLS-to-Patch Attention
Attention matrices were extracted from the final (12th) transformer block. The tensor was averaged across the 3 heads, and the vector representing the CLS token's attention toward the 196 spatial patches was isolated, reshaped to 14×14, upsampled to 224×224, and mapped onto the original image.

### 12.3 Attention Visualization Results
A total of 10 attention maps (5 correct classifications, 5 incorrect classifications) were generated, structurally characterizing the physical lesion boundaries prioritized by the self-attention mechanism.

### 12.4 Interpretation Limitations
These maps are qualitative artifacts of transformer computation (CLS-token transformer attention visualization) and do not constitute clinically verified diagnostic explanations or lesion segmentations.

## 13. Computational Resource Analysis

### 13.1 Training Time
The 10-epoch training required 2726.7 seconds (45.4 minutes).

### 13.2 GPU
Training was executed on an NVIDIA GeForce RTX 4050 Laptop GPU.

### 13.3 Parameter Count
The ViT-Tiny model possesses 5,525,767 trainable parameters.

### 13.4 Feature Dimension
The network terminates in a 192-dimensional latent feature vector.

## 14. Comparison With Earlier Baselines
The following table summarizes the three trained classical baselines. 

| Experiment | Accuracy | Macro F1 | Weighted F1 | Parameters |
|---|---:|---:|---:|---:|
| EXP-001 ResNet18 | 0.8030 | 0.6705 | 0.8083 | 11,180,103 |
| EXP-002 EfficientNet-B0 | 0.8079 | 0.7160 | 0.8157 | 4,016,423 |
| EXP-003 ViT-Tiny | 0.8055 | 0.7231 | 0.8114 | 5,525,767 |

EXP-003 produced a macro F1 of 0.7231, compared with 0.7160 for EXP-002 and 0.6705 for EXP-001 under their respective experimental protocols. Accuracy and Weighted F1 were comparably stable across the architectures. 

## 15. Limitations
Several limitations must be addressed:
- The stark class imbalance suppressed predictive reliability for critical pathologies (MEL, AKIEC) due to small minority-class support.
- Performance relies on a single static validation split; no external testing or clinical evaluation occurred.
- The ImageNet-21K pretraining utilized for ViT-Tiny introduces a discrepancy when directly evaluating architectural advantages against the ImageNet-1K pretrained EXP-001 and EXP-002.
- Extracted attention maps are strictly qualitative.
- Computational and dataset constraints define the upper limit of achievable convergence.
- No quantum capabilities or advantages are measured within this phase of classical baseline establishment.

## 16. Reproducibility Information
- Seed: 42
- Dataset Split: Lesion-level, zero overlap (7,974 train / 2,041 val)
- Model: `vit_tiny_patch16_224.augreg_in21k_ft_in1k` (timm)
- Image Size: 224×224
- Optimizer: Adam
- Learning Rate: 1e-4 (Fixed)
- Batch Size: 32
- Epochs: 10
- Loss: Weighted Cross-Entropy
- Preprocessing: Resize, ToTensor, ImageNet Normalization
- Checkpoint Criterion: Lowest Validation Loss
- GPU: NVIDIA RTX 4050 Laptop GPU

## 17. Conclusion
EXP-003 successfully trained and validated a ViT-Tiny classical baseline. Achieving a Macro F1 of 0.7231 and an Accuracy of 0.8055, the Vision Transformer provides a distinct and competitive self-attention pathway yielding a 192-dimensional feature embedding. Having completed EXP-001, EXP-002, and EXP-003, the research foundation is fully established, enabling the forthcoming investigations into multi-model dimensionality reduction and hybrid quantum-classical feature fusion.
