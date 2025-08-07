# MLP, GMM, PCA, AE & VAE from Scratch

## 📌 Overview

This assignment covers a wide range of foundational machine learning concepts. The major components include implementing models from scratch (MLP, GMM, PCA), and applying deep learning frameworks (PyTorch) for AutoEncoders and Variational AutoEncoders. Each task simulates a real-world application with in-depth experimentation and evaluation.

---

## 📁 Contents

### 🔹 1. Multi-Layer Perceptron (MLP) – From Scratch
Implementations of a configurable MLP model applied to three problems:

- **Symbol Classification (Multi-Class)**  
  Classifies handwritten historical symbols using a custom-built MLP with support for different activations and optimizers. 10-fold validation included.

- **Bangalore Housing Price Prediction (Regression)**  
  Regression MLP built from scratch to estimate real estate prices using cleaned and normalized real-world data.

- **Multi-Label News Article Classification**  
  TF-IDF feature extraction and label binarization for multi-topic classification. Includes hyperparameter tuning and performance evaluation.

Each MLP supports:
- Sigmoid, Tanh, ReLU (custom)
- SGD, Batch GD, Mini-Batch GD (custom)
- Fully configurable architecture and hyperparameters

---

### 🔹 2. Gaussian Mixture Model (GMM) – From Scratch

- Implemented EM-based GMM to model pixel intensities.
- Segmentation of MRI brain scan into White Matter, Gray Matter, and CSF.
- Visualized GMM distributions and reported segmentation accuracy.
- Analyzed misclassification patterns using frequency vs. intensity plots.

---

### 🔹 3. Principal Component Analysis (PCA) – From Scratch

- PCA implemented via covariance matrix and eigen decomposition.
- Applied to MNIST for:
  - Dimensionality reduction (500, 300, 150, 30)
  - Lossy reconstruction and visualization
  - Classification performance with vs. without PCA
- Observations reported on effectiveness of PCA in real-world contexts.

---

### 🔹 4. AutoEncoder (AE) – Using PyTorch

- Built encoder–decoder architecture for MNIST anomaly detection.
- Normal digit = last digit of roll number; all others treated as anomalies.
- Reconstruction error histogram for normal vs. anomaly.
- Evaluated with Precision, Recall, F1-score, and ROC-AUC for different latent sizes.

---

### 🔹 5. Variational AutoEncoder (VAE) – Using PyTorch

- Trained VAE on MNIST with Binary Cross Entropy loss.
- Explored:
  - Effect of removing reconstruction / KL loss
  - Latent space visualization with 2D Gaussian sampling
  - Comparison with MSE loss
- Included generated grid reconstructions and comprehensive commentary.

---
