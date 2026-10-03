# EXP-003 Results Summary

## Overall Performance
The ViT-Tiny/16 classical baseline (EXP-003) achieved its optimal checkpoint at Epoch 8 with a validation loss of 0.550898. The model reported an overall Accuracy of 0.8055. Because the HAM10000 dataset is significantly imbalanced, macro and weighted metrics are critical for assessing true classifier performance. The model achieved a Macro F1 score of 0.7231 and a Weighted F1 score of 0.8114 on the validation set.

## Per-Class Performance
Performance varied widely across the 7 diagnostic classes, primarily correlated with class representation in the training data despite the application of Weighted Cross-Entropy Loss:

- **AKIEC**: F1 = 0.57 (Support: 68)
- **BCC**: F1 = 0.78 (Support: 132)
- **BKL**: F1 = 0.68 (Support: 244)
- **DF**: F1 = 0.68 (Support: 34)
- **MEL**: F1 = 0.58 (Support: 225)
- **NV**: F1 = 0.89 (Support: 1302)
- **VASC**: F1 = 0.88 (Support: 36)

The majority class (Melanocytic Nevi, NV) demonstrated the highest F1 score (0.89), while minority and complex classes such as Melanoma (MEL, 0.58) and Actinic Keratoses (AKIEC, 0.57) exhibited relatively lower predictive performance.

## Confusion Matrix Observations
Analysis of the raw and normalized confusion matrices revealed systemic classification challenges typical of dermatoscopic datasets. The model often confused Melanoma (MEL) with Benign Keratosis-like lesions (BKL) and Nevi (NV). While class weighting improved the recall for minority classes compared to unweighted baselines, precision suffered in categories like AKIEC and MEL, where the model was more prone to false positive assignments originating from the NV class. 

## Resource Results
The experiment utilized 5,525,767 trainable parameters to extract a 192-dimensional feature embedding. Training spanned exactly 10 epochs with a batch size of 32. The process required 2726.7 seconds (approximately 45.4 minutes) utilizing an NVIDIA GeForce RTX 4050 Laptop GPU.

## Attention Visualization
A post-training visualization pass was executed to generate 10 qualitative attention maps (5 correct, 5 incorrect predictions). Extracting the CLS-token-to-patch attention weights from the final transformer block yielded detailed 14×14 grid overlays. These visual artifacts verified that the model's self-attention mechanism localized distinct structural features within the lesion images, confirming the architectural integrity of the ViT implementation.

## Comparison With EXP-001 and EXP-002
In comparison to preceding CNN baselines under identical strict split protocols:

- **EXP-001 (ResNet18)**: Accuracy = 0.8030, Macro F1 = 0.6705, Weighted F1 = 0.8083, Parameters = 11,180,103
- **EXP-002 (EfficientNet-B0)**: Accuracy = 0.8079, Macro F1 = 0.7160, Weighted F1 = 0.8157, Parameters = 4,016,423
- **EXP-003 (ViT-Tiny)**: Accuracy = 0.8055, Macro F1 = 0.7231, Weighted F1 = 0.8114, Parameters = 5,525,767

EXP-003 produced the highest Macro F1 (0.7231), indicating slightly more equitable performance across the imbalanced classes, while maintaining comparable Accuracy and Weighted F1 metrics to EfficientNet-B0. Note that ViT-Tiny utilized an ImageNet-21K pretraining regime, differing from the ImageNet-1K pretraining of EXP-001 and EXP-002.

## Interpretation
The ViT-Tiny architecture serves as an effective self-attention baseline for HAM10000 lesion classification. The 192-dimensional representation demonstrated robust discriminative capacity. The results substantiate ViT-Tiny as a viable and competitive feature extractor alongside CNN backbones, validating its inclusion in the forthcoming hybrid quantum-classical feature fusion architecture.

## Limitations
Several limitations are inherent to this experiment:
- The severe class imbalance restricts performance on critical minority classes (e.g., MEL and AKIEC).
- Validation was restricted to the 2,041-image static split without external dataset evaluation or clinical testing.
- The attention maps remain purely qualitative and do not constitute a clinically verified diagnostic explanation.
- The architectural pretraining (ImageNet-21K) differs from the CNN baselines (ImageNet-1K), slightly confounding direct architectural comparison.
- No quantum advantage is assessed or claimed at this purely classical baseline stage.

## Research Implication
EXP-003 successfully establishes a third, distinct classical feature extraction pathway (transformer-based, 192-D). With ResNet18 (512-D), EfficientNet-B0 (1280-D), and ViT-Tiny (192-D) completely defined and trained, the project is now positioned to investigate multi-model feature fusion, dimensionality reduction (PCA), and the integration of Variational Quantum Circuits (VQC) in subsequent experimental phases.
