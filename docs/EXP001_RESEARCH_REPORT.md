# Resource-Efficient and Noise-Aware Hybrid Quantum-Classical Learning for Multiclass Skin Lesion Classification

## Abstract

Multiclass skin lesion classification is a critical challenge in automated dermatological diagnosis, often hindered by severe class imbalance and high computational resource requirements. While deep neural networks have achieved strong performance, they typically rely on large architectures that are computationally expensive. This project investigates resource-efficient methodologies by exploring a hybrid quantum-classical architecture capable of robust predictive performance under constrained conditions. As a necessary first step, this report establishes a reproducible classical baseline using the HAM10000 dataset. We utilize a ResNet18 model fine-tuned on a strict lesion-level split to prevent data leakage, and handle class imbalance using training-only balanced class weights. The classical baseline achieves a weighted F1-score of 80.83% and an overall accuracy of 80.30%. This firmly established baseline serves as the reference point for our planned subsequent investigations into lightweight feature extraction and shallow variational quantum circuits.

---

# 1. INTRODUCTION

Skin lesion classification is a challenging image classification problem due to the high visual similarity between different skin diseases, varying lesion sizes, and significant variability in skin types. The multiclass classification challenge is further complicated by severe class imbalance, as seen in the HAM10000 dataset, where benign melanocytic nevi heavily outnumber critical malignancies like melanoma. 

While modern deep learning architectures address these challenges effectively, they are often computationally demanding, prompting the need for more resource-efficient models. Hybrid quantum-classical learning has recently emerged as a novel approach that potentially leverages high-dimensional feature spaces using constrained quantum resources. However, before introducing quantum components, a strong and reproducible classical baseline must be rigorously established. This report details the establishment of a robust classical baseline that will serve as the comparative foundation for evaluating the resource efficiency and noise awareness of future hybrid quantum models.

---

# 2. PROBLEM STATEMENT

The primary problem is to achieve resource-efficient multiclass skin lesion classification while investigating whether a carefully designed hybrid quantum-classical architecture can provide useful predictive performance under constrained quantum resources.

---

# 3. OBJECTIVES

The overall project encompasses several objectives:

1. Establish a reliable classical baseline. **(COMPLETED)**
2. Investigate lightweight classical feature extraction. *(FUTURE WORK)*
3. Reduce feature dimensionality before quantum processing. *(FUTURE WORK)*
4. Investigate shallow quantum circuits. *(FUTURE WORK)*
5. Study the effect of limited qubit counts. *(FUTURE WORK)*
6. Study noise robustness. *(FUTURE WORK)*
7. Compare hybrid models with the classical baseline. *(FUTURE WORK)*
8. Analyze the trade-off between predictive performance and computational/quantum resources. *(FUTURE WORK)*

---

# 4. DATASET

The dataset utilized is HAM10000, which contains dermatoscopic images across 7 classes.

Total images = 10,015

Classes:

| Class | Count |
| ----- | ----: |
| AKIEC |   327 |
| BCC   |   514 |
| BKL   | 1,099 |
| DF    |   115 |
| MEL   | 1,113 |
| NV    | 6,705 |
| VASC  |   142 |

The dataset exhibits a severe class imbalance, with the NV class dominating the distribution. Furthermore, the dataset contains multiple images for the same underlying physical lesion. To prevent data leakage and ensure an unbiased evaluation, the project employs a strict lesion-level splitting strategy rather than a naive image-level split.

---

# 5. DATA PREPARATION AND SPLITTING

Data loading is metadata-driven. The dataset was split into training and validation sets using a lesion-level strategy with random seed 42.

* 7,974 training images
* 2,041 validation images
* 5,976 unique training lesions
* 1,494 unique validation lesions
* zero lesion overlap

Lesion-level splitting is critical because multiple images of the same lesion share highly correlated visual features. Allowing the same lesion to appear in both training and validation sets would artificially inflate validation performance metrics and fail to represent true generalization capability.

---

# 6. CLASS IMBALANCE HANDLING

To mitigate the severe class imbalance, weighted CrossEntropyLoss was employed. Class weights were computed using an inverse-frequency balanced strategy. Importantly, to prevent data leakage, these weights were calculated strictly from the training split labels, completely excluding validation label frequencies.

---

# 7. CLASSICAL BASELINE MODEL

The classical baseline model is ResNet18. It was initialized with ImageNet pretrained weights and subjected to full fine-tuning. The final classification layer was replaced with a linear layer mapping 512 features to the 7 target classes. 

ResNet18 was selected as the baseline because it is an established CNN architecture that is relatively lightweight compared to larger networks like ResNet50 or EfficientNet. This makes it highly suitable as a feature extraction backbone and a practical reference point for later resource-efficiency comparisons in hybrid architectures.

---

# 8. TRAINING CONFIGURATION

| Parameter | Value |
|-----------|-------|
| Model | ResNet18 |
| Pretraining | ImageNet |
| Input | 224×224 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Epochs | 10 |
| Seed | 42 |
| Loss | Weighted CrossEntropyLoss |
| W&B | Disabled |
| Best checkpoint criterion | Lowest validation loss |
| GPU | NVIDIA RTX 4050 Laptop GPU, 6 GB |

Training augmentations applied: RandomResizedCrop(224, scale=(0.8,1.0)), RandomHorizontalFlip(0.5), RandomRotation(20), ColorJitter(0.1,0.1,0.1), ToTensor(), and ImageNet normalization.

---

# 9. EXP-001 RESULTS

| Metric | Result |
|--------|--------|
| Accuracy | 80.30% |
| Macro Precision | 65.16% |
| Macro Recall | 70.95% |
| Macro F1 | 67.05% |
| Weighted F1 | 80.83% |
| Best Epoch | 7 |
| Best Validation Loss | 0.5936 |
| Training Time | 3316.6 seconds |

The discrepancy between the weighted F1-score (80.83%) and the macro F1-score (67.05%) reflects the significant impact of class imbalance. While the model performs well on the majority classes, minority classes pull down the unweighted macro average. 

---

# 10. PER-CLASS RESULTS

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| AKIEC | 0.4037 | 0.6471 | 0.4972 | 68.0 |
| BCC | 0.7020 | 0.8030 | 0.7491 | 132.0 |
| BKL | 0.6429 | 0.6639 | 0.6532 | 244.0 |
| DF | 0.6250 | 0.4412 | 0.5172 | 34.0 |
| MEL | 0.5714 | 0.5867 | 0.5789 | 225.0 |
| NV | 0.9363 | 0.8802 | 0.9074 | 1302.0 |
| VASC | 0.6800 | 0.9444 | 0.7907 | 36.0 |

---

# 11. CONFUSION MATRIX ANALYSIS

The confusion matrix indicates that the model heavily favors correctly identifying the NV class, which has the highest support. Despite the application of class weights, precision for the AKIEC class remains low, meaning the model suffers from false positives when predicting AKIEC. However, the recall for VASC and BCC is notably strong, suggesting that the model is relatively successful at identifying true positives within these specific minority classes when they occur.

---

# 12. GRAD-CAM ANALYSIS

Grad-CAM was generated from the last convolutional layer (`model.layer4[-1]`) on sample validation images. Grad-CAM is utilized for qualitative interpretability to observe whether the model's visual attention correlates with the actual lesion region rather than background artifacts. It is not intended to provide clinical interpretability or diagnostic validity, but rather to ensure the network is learning meaningful spatial features before progressing to quantum feature encodings.

---

# 13. RESOURCE PROFILE

* GPU: NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM
* Training time: 3316.6 seconds (approximately 55.3 minutes)

Parameter count, memory usage, FLOPs, and inference latency were not measured in EXP-001.

---

# 14. CURRENT LIMITATIONS

* This report establishes a classical baseline only.
* No quantum model has yet been evaluated.
* No noise simulation has yet been evaluated.
* No quantum advantage is claimed.
* Only train/validation results are currently available.
* Severe class imbalance remains challenging, particularly for minority class precision.
* Baseline performance should not be interpreted as clinical performance.
* There has been no external or prospective clinical validation.
* The model makes no claim of deployment readiness.

---

# 15. PROPOSED HYBRID QUANTUM-CLASSICAL PIPELINE

**Proposed Future Experimental Pipeline**

HAM10000 images
↓
Lightweight CNN feature extractor
↓
Feature vector
↓
Dimensionality reduction / feature selection
↓
Low-dimensional quantum feature vector
↓
Quantum encoding
↓
Shallow variational quantum circuit
↓
Measurement
↓
7-class classifier

Future experiments will involve extracting features via CNNs, employing PCA or selection techniques, and feeding representations into varying qubit architectures (e.g., 4, 6, 8 qubits). We will explore angle encoding, shallow Variational Quantum Circuits (VQCs) of different depths, simulated quantum noise, and noise-aware training techniques, ultimately comparing these hybrid models with the classical baseline.

---

# 16. PROPOSED EXPERIMENT MATRIX

| Experiment | Description | Status |
|---|---|---|
| EXP-001 | ResNet18 classical baseline | COMPLETED |
| EXP-002 | Classical feature extraction + PCA | PLANNED |
| EXP-003 | Hybrid quantum-classical baseline | PLANNED |
| EXP-004 | Qubit/resource comparison | PLANNED |
| EXP-005 | Noise robustness study | PLANNED |
| EXP-006 | Noise-aware hybrid model | PLANNED |

---

# 17. RESEARCH QUESTIONS

* RQ1: Can a lightweight classical feature extractor provide sufficiently informative low-dimensional representations for hybrid quantum classification?
* RQ2: How does reducing the feature dimension affect hybrid model performance?
* RQ3: How does the number of qubits affect classification performance and resource requirements?
* RQ4: How does simulated quantum noise affect classification performance?
* RQ5: Can noise-aware training improve robustness under simulated noise?
* RQ6: What trade-off exists between predictive performance and quantum resource usage?

---

# 18. EVALUATION METRICS

Future experiments will consistently report Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, Per-class metrics, and Confusion matrices. 

For resource efficiency, experiments will report the number of qubits, circuit depth, number of trainable quantum parameters, number of quantum gates, classical feature dimension, training time, and inference time where measurable. For noise experiments, noiseless performance, noisy performance, performance degradation, and robustness across noise levels will be recorded.

---

# 19. REPRODUCIBILITY

* Operating System: win32
* Python version: 3.14.7 (tags/v3.14.7:823f032)
* PyTorch version: 2.11.0+cu128
* torchvision version: 0.26.0+cu128
* CUDA version: 12.8
* GPU: NVIDIA GeForce RTX 4050 Laptop GPU
* Seed: 42
* Dataset configuration: `data_dir` pointing directly to the LFS downloaded directory containing class subfolders.
* Experiment ID: EXP-001
* Checkpoint filename: `best_model.pth`
* Output directory: `results\baseline_resnet18`

---

# 20. CURRENT CONTRIBUTION

The completed contribution at this stage is the establishment of a reproducible lesion-level classical baseline and an evaluation framework that can serve as the reference point for subsequent resource-constrained hybrid quantum-classical experiments.

---

# 21. CONCLUSION

The HAM10000 dataset was successfully prepared and subjected to a lesion-level leakage-free split. The severe class imbalance was handled using training-only class weights. The ResNet18 classical baseline was completed, achieving a macro F1 of 67.05% and a weighted F1 of 80.83%. Grad-CAM and evaluation artifacts were generated successfully. This firmly establishes a classical reference point intended for future comparative evaluations as the project introduces dimensionality reduction and hybrid quantum components.

---

# 22. REFERENCES

**References to be added during literature review**

* HAM10000 dataset
* ResNet
* transfer learning
* class imbalance
* Grad-CAM
* quantum machine learning
* variational quantum circuits
* quantum feature encoding
* quantum noise / noisy simulation

---

## Recommended Figures

1. Overall proposed hybrid quantum-classical pipeline — FUTURE
2. HAM10000 class distribution — CURRENT (To be generated from data)
3. Training/validation loss curves — CURRENT (Available in training_history.json)
4. Training/validation accuracy curves — CURRENT (Available in training_history.json)
5. Confusion matrix — CURRENT (Exists: confusion_matrix.png)
6. Normalized confusion matrix — CURRENT (Exists: confusion_matrix_normalized.png)
7. Grad-CAM examples — CURRENT (Exists in gradcam/ directory)
8. Future quantum circuit architecture — FUTURE

## Recommended Tables

1. Dataset class distribution
2. EXP-001 training configuration
3. EXP-001 overall metrics
4. EXP-001 per-class metrics
5. Future experiment matrix
