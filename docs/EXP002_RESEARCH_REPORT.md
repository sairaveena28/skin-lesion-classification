# EXP-002 Research Report

## EfficientNet-B0 Classical Baseline

### 1. Experiment Overview

* **Experiment ID:** EXP-002
* **Model:** EfficientNet-B0

This experiment investigates whether a lightweight, modern Convolutional Neural Network (CNN) architecture provides a different predictive accuracy, class-balance handling, and computational resource profile compared to the ResNet18 baseline (EXP-001). This experiment is part of a larger research progression toward hybrid quantum-classical learning. By keeping the dataset split and core training protocol strictly controlled, EXP-002 serves to isolate the effect of the architectural change (ResNet18 vs. EfficientNet-B0) on performance. 

This experiment is not intended to claim that EfficientNet-B0 is the final or superior model universally, but rather to evaluate its behavior under constrained conditions as a secondary classical baseline.

---

### 2. Research Objective

The primary scientific objective is to evaluate EfficientNet-B0 under the exact same controlled HAM10000 experimental settings used for EXP-001. By doing so, we establish a second reliable classical deep-learning reference point before advancing to subsequent phases of the project (e.g., lightweight feature extraction, dimensionality reduction, and hybrid quantum-classical experiments).

This experiment is intended to provide empirical evidence to inform subsequent model and feature selection strategies for the quantum-classical pipeline, rather than functioning as the final optimized model.

---

### 3. Dataset

The dataset utilized is HAM10000, consisting of dermatoscopic skin lesion images.

* **Total images:** 10,015
* **Number of classes:** 7

**Classes & Original Distribution:**
* AKIEC = 327
* BCC = 514
* BKL = 1,099
* DF = 115
* MEL = 1,113
* NV = 6,705
* VASC = 142

The HAM10000 dataset exhibits severe class imbalance, with the benign melanocytic nevi (NV) class dominating the distribution. Due to this heavy skew, evaluating macro-level metrics (Macro F1, Precision, and Recall) is critically important to accurately interpret the model's ability to classify rare and potentially malignant lesions without being biased by majority class accuracy.

---

### 4. Data Splitting and Leakage Prevention

Data loading is metadata-driven. The dataset was split into training and validation sets using a strict lesion-level strategy with random seed 42.

* **Training samples:** 7,974
* **Validation samples:** 2,041
* **Seed:** 42
* **Lesion overlap:** 0

Lesion-level splitting ensures that multiple augmented or diverse images of the same physical lesion do not straddle the training and validation sets. A lesion overlap of zero is crucial to prevent data leakage and ensure that validation metrics reflect the model's ability to generalize to unseen lesions, rather than merely recognizing specific, previously seen lesions. This splitting strategy does not definitively prove universal generalization, but is required for valid internal comparison.

---

### 5. Image Preprocessing and Augmentation

The preprocessing and augmentation pipelines are implemented to match EXP-001 identically, ensuring a controlled comparison.

**Training:**
* RandomResizedCrop(224, scale=(0.8, 1.0))
* RandomHorizontalFlip(0.5)
* RandomRotation(20)
* ColorJitter(0.1, 0.1, 0.1)
* ToTensor
* ImageNet normalization

**Validation:**
* Resize(224, 224)
* ToTensor
* ImageNet normalization

No extra preprocessing or augmentations were introduced to prevent obfuscating the architectural differences between EXP-001 and EXP-002.

---

### 6. EfficientNet-B0 Architecture

The chosen architecture is `torchvision`'s implementation of EfficientNet-B0.
* **Initialization:** Pretrained ImageNet weights (`IMAGENET1K_V1`).
* **Classifier:** The final linear classification layer was adapted to output exactly 7 classes for the HAM10000 dataset.
* **Transfer Learning:** The model underwent full fine-tuning. There was no frozen-backbone stage, no two-stage transfer learning, and no differential learning rates applied across layers.

EfficientNet-B0 is highly relevant to a resource-aware research direction due to its compound scaling design, achieving a strong balance of parameter efficiency and accuracy. This baseline provides an alternative lightweight feature extractor for the quantum-classical pipeline. We make no unsupported claims about its absolute optimality in this domain.

---

### 7. Training Configuration

* **Epochs:** 10
* **Batch size:** 32
* **Optimizer:** Adam
* **Learning rate:** 0.0001
* **Random seed:** 42
* **W&B:** Disabled
* **Scheduler:** None
* **Early stopping:** None
* **TTA:** None
* **MixUp:** None
* **CutMix:** None
* **Oversampling:** None
* **Undersampling:** None
* **WeightedRandomSampler:** None
* **SMOTE:** None

---

### 8. Class Imbalance Handling

To mitigate the severe class imbalance, the experiment utilized a class-weighted `CrossEntropyLoss`. 

* Class weights were calculated using **ONLY** the label frequencies in the training split.
* The validation data was strictly excluded from class-weight calculations to preserve a zero-knowledge validation set.
* No resampling techniques (like WeightedRandomSampler, SMOTE, oversampling, or undersampling) were used.

This strategy was kept strictly consistent with EXP-001 to ensure that any observed performance differences originate from the EfficientNet-B0 architecture rather than varied data distribution methods.

---

### 9. EXP-002 RESULTS

| Metric | Result |
|--------|--------|
| Accuracy | 80.79% |
| Macro Precision | 70.71% |
| Macro Recall | 73.62% |
| Macro F1 | 71.60% |
| Weighted F1 | 81.57% |
| Best Epoch | 9 |
| Best Validation Loss | 0.5550 |
| Training Time | 2633.5 seconds |

EfficientNet-B0 demonstrated improvements across macro metrics compared to EXP-001 (ResNet18), particularly noting an uplift in the Macro F1 score (71.60% vs 67.05%), indicating a slightly more robust handling of minority classes under identical class-weighting conditions. 

---

### 10. PER-CLASS RESULTS

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| AKIEC | 0.5278 | 0.5588 | 0.5429 | 68.0 |
| BCC | 0.7939 | 0.7879 | 0.7909 | 132.0 |
| BKL | 0.6891 | 0.6721 | 0.6805 | 244.0 |
| DF | 0.7500 | 0.6176 | 0.6774 | 34.0 |
| MEL | 0.5000 | 0.7378 | 0.5961 | 225.0 |
| NV | 0.9390 | 0.8625 | 0.8991 | 1302.0 |
| VASC | 0.7500 | 0.9167 | 0.8250 | 36.0 |

---

### 11. CONFUSION MATRIX ANALYSIS

The generated confusion matrices (`confusion_matrix.png` and `confusion_matrix_normalized.png`) reflect performance distributions across the classes. While the NV class (benign melanocytic nevi) maintains very high precision and recall, EfficientNet-B0 shows an ability to capture VASC and BCC lesions with notably high recall and precision. Minority classes like AKIEC and MEL remain challenging, exhibiting high false positives/negatives, underscoring the intrinsic difficulty of these morphologies under standard imbalanced learning conditions. 

---

### 12. GRAD-CAM ANALYSIS

Grad-CAM outputs were generated from the last convolutional layer (`model.features[-1]`) on sample validation images. These visualizations serve to confirm that the model's spatial attention appropriately aligns with the visible lesions rather than spurious background artifacts. This qualitative check is vital for ensuring the feature extraction backbone produces meaningful spatial representations before these features are utilized in dimensionality reduction and quantum feature encodings. 

---

### 13. RESOURCE PROFILE

* **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB VRAM
* **Training time:** 2633.5 seconds (approximately 43.9 minutes)

Compared to EXP-001 (3316.6 seconds), EfficientNet-B0 completed the 10-epoch training cycle approximately 11.4 minutes faster in this environment, marking an observable advantage in computational resource efficiency for this phase of the pipeline.

---

### 14. CURRENT LIMITATIONS

* This report establishes a secondary classical baseline.
* No quantum model or noise simulation has been evaluated yet.
* No quantum advantage is claimed.
* Performance remains bottlenecked by severe class imbalance, particularly affecting minority class precision.
* The reported metrics reflect internal validation performance only and should not be interpreted as prospective clinical validation or diagnostic readiness.

---

### 15. PROPOSED HYBRID QUANTUM-CLASSICAL PIPELINE

**Proposed Future Experimental Pipeline**

HAM10000 images
↓
Lightweight CNN feature extractor (ResNet18 / EfficientNet-B0)
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

With EXP-001 and EXP-002 complete, the focus will shift toward feature extraction, PCA/selection techniques, and configuring variational quantum circuits (VQCs) to ingest low-dimensional feature vectors.

---

### 16. PROPOSED EXPERIMENT MATRIX

| Experiment | Description | Status |
|---|---|---|
| EXP-001 | ResNet18 classical baseline | COMPLETED |
| EXP-002 | EfficientNet-B0 classical baseline | COMPLETED |
| EXP-003 | Classical feature extraction + PCA | PLANNED |
| EXP-004 | Hybrid quantum-classical baseline | PLANNED |
| EXP-005 | Qubit/resource comparison | PLANNED |
| EXP-006 | Noise robustness study | PLANNED |
| EXP-007 | Noise-aware hybrid model | PLANNED |

---

### 17. RESEARCH QUESTIONS

* **RQ1:** Can a lightweight classical feature extractor (like EfficientNet-B0) provide sufficiently informative low-dimensional representations for hybrid quantum classification?
* **RQ2:** How does reducing the feature dimension affect hybrid model performance?
* **RQ3:** How does the number of qubits affect classification performance and resource requirements?
* **RQ4:** How does simulated quantum noise affect classification performance?
* **RQ5:** Can noise-aware training improve robustness under simulated noise?
* **RQ6:** What trade-off exists between predictive performance and quantum resource usage?

---

### 18. EVALUATION METRICS

Future experiments will continue to report Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, Per-class metrics, and Confusion matrices. 

Resource efficiency metrics will expand to include the number of qubits, circuit depth, trainable quantum parameters, classical feature dimensions, and respective training/inference times.

---

### 19. REPRODUCIBILITY

* **Operating System:** win32
* **Python version:** 3.14.7
* **PyTorch version:** 2.11.0+cu128
* **CUDA version:** 12.8
* **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU
* **Seed:** 42
* **Dataset configuration:** `data_dir` pointing to the LFS downloaded directory containing class subfolders.
* **Experiment ID:** EXP-002
* **Config path:** `configs/efficientnet_b0.yaml`
* **Checkpoint filename:** `best_model.pth`
* **Output directory:** `results\efficientnet_b0\`

---

### 20. CURRENT CONTRIBUTION

The current contribution is the robust establishment of a secondary, highly efficient classical baseline (EfficientNet-B0) on the HAM10000 dataset using a strict lesion-level split and controlled training conditions. This adds a critical comparative dimension for future hybrid quantum-classical resource evaluations.

---

### 21. CONCLUSION

The EfficientNet-B0 classical baseline (EXP-002) was successfully trained and evaluated under identically controlled conditions as EXP-001. The model achieved a Macro F1 of 71.60% and a Weighted F1 of 81.57% while utilizing fewer computational resources (2633.5s training time). These results establish EfficientNet-B0 as a viable and potentially more efficient candidate for the CNN feature extraction backbone in the upcoming quantum-classical architecture experiments.

---

### 22. REFERENCES

**References to be added during literature review**

* HAM10000 dataset
* EfficientNet architecture
* transfer learning
* class imbalance
* Grad-CAM
* quantum machine learning
* variational quantum circuits
* quantum feature encoding
* quantum noise / noisy simulation
