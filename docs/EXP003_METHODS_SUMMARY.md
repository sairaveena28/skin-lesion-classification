# EXP-003 Methods Summary

## Objective
The objective of EXP-003 was to establish a highly reproducible, robust, and controlled classical baseline using the Vision Transformer (ViT-Tiny/16) architecture for multiclass skin lesion classification. This provides a baseline feature extractor representing a self-attention paradigm, which will be compared and fused with CNN baselines in forthcoming hybrid quantum-classical experiments.

## Dataset
The experiment utilized the HAM10000 dataset, containing 10,015 dermatoscopic images across 7 diagnostic categories: Actinic keratoses and intraepithelial carcinoma (AKIEC), basal cell carcinoma (BCC), benign keratosis-like lesions (BKL), dermatofibroma (DF), melanoma (MEL), melanocytic nevi (NV), and vascular lesions (VASC).

## Data Split
To prevent data leakage, a strict lesion-level split was enforced based on the unique `lesion_id` associated with the images. The dataset was divided into a training set of 7,974 images (5,976 unique lesions) and a validation set of 2,041 images (1,494 unique lesions), ensuring zero lesion overlap between the two sets. A fixed random seed (42) was used to guarantee reproducibility.

## Model
The selected model was `vit_tiny_patch16_224.augreg_in21k_ft_in1k` (ViT-Tiny/16) provided by the `timm` library. The architecture consists of 12 transformer blocks, 3 attention heads, a patch size of 16×16, and an embedding dimension of 192. A linear classification head was adapted to output the 7 skin lesion classes. The model contained 5,525,767 total parameters, all of which were fully fine-tuned during training.

## Preprocessing
Validation images were resized to 224×224 pixels, converted to PyTorch tensors, and normalized using the standard ImageNet mean ([0.485, 0.456, 0.406]) and standard deviation ([0.229, 0.224, 0.225]).

## Augmentation
During training, the following data augmentation transformations were applied to mitigate overfitting and improve generalization:
- Random Resized Crop (224×224, scale 0.8 to 1.0)
- Random Horizontal Flip (p=0.5)
- Random Rotation (±20 degrees)
- Color Jitter (brightness=0.1, contrast=0.1, saturation=0.1)

## Loss Function
To address the severe class imbalance inherent in the HAM10000 dataset (e.g., NV containing significantly more samples than DF or VASC), a Weighted Cross-Entropy Loss was employed. The class weights were computed inversely proportional to class frequencies exclusively from the training split, preventing any validation statistics from influencing the optimization objective.

## Optimization
The model was optimized using the Adam optimizer. The learning rate was set to a constant 1e-4 without the use of a learning rate scheduler, enabling a direct and unembellished assessment of the baseline architecture's learning capacity.

## Training Protocol
The training spanned exactly 10 epochs with a batch size of 32. Advanced sampling and regularization techniques—such as Test-Time Augmentation (TTA), MixUp, CutMix, SMOTE, and class-specific oversampling/undersampling—were strictly excluded to preserve the integrity and simplicity of the baseline comparison.

## Checkpoint Selection
The final model was selected solely based on the lowest validation loss achieved across the 10 epochs. Evaluation metrics such as accuracy or F1-score were not used as criteria for checkpoint selection.

## Evaluation
The selected checkpoint was evaluated on the frozen validation set. The reported metrics encompass Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted Precision, Weighted Recall, Weighted F1, and a detailed per-class classification report (Precision, Recall, F1, Support). Both raw and normalized confusion matrices were generated for categorical performance analysis.

## Attention Visualization
A post-training visualization was conducted to qualitatively assess the ViT-Tiny's attention mechanism. The CLS-token transformer attention was extracted from the final transformer block, averaged across attention heads, and mapped onto the 14×14 patch grid. The resulting matrix was upsampled to 224×224 using bilinear interpolation and overlaid onto the original images using a heatmap. This produced 5 correctly predicted and 5 incorrectly predicted attention map samples. 

## Reproducibility
The entire experimental pipeline is constrained by deterministic procedures, a fixed initialization seed (42), and the exclusion of stochastic post-processing or dynamic sampling. Checkpoints, configuration files, and detailed execution logs were captured to ensure the strict reproducibility of the reported results.

## Key Methodological Notes
It is vital to note that the ViT-Tiny model was pretrained on ImageNet-21K followed by fine-tuning on ImageNet-1K (`augreg_in21k_ft_in1k`). This represents a broader pretraining regime compared to EXP-001 (ResNet18) and EXP-002 (EfficientNet-B0), which utilized ImageNet-1K exclusively. This discrepancy is a documented methodological difference characteristic of standard ViT checkpoints rather than an error in experimental design.
