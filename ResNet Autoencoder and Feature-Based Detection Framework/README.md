# ResNet Autoencoder and Feature-Based Detection Framework

## Overview

This project implements a two-stage framework for detecting whether specific real images were used in the training of generative models. The system combines:

1. **ResNet-based Convolutional Autoencoder** for learning deep latent representations from synthetic images.
2. **Feature-Based Classification Pipeline** using both learned and handcrafted descriptors to classify real images as used or unused.

---

## 1. ResNet Autoencoder

We utilize a deep convolutional autoencoder with residual connections to ensure stable training and robust feature extraction. The encoder consists of four Conv2D layers with increasing filter sizes (32 → 64 → 128 → 256), each followed by a residual block that includes:

- Two 3×3 Conv2D layers
- Batch Normalization
- LeakyReLU activation
- Shortcut (skip) connections

The encoder outputs a **512-dimensional latent vector** via a GlobalAveragePooling2D layer and a fully connected Dense layer.

The decoder reconstructs the image using:
- A Dense layer reshaped to 16×16×128
- Three Conv2DTranspose layers (128 → 64 → 32), each with residual blocks
- A final Conv2DTranspose layer with 3 filters and sigmoid activation

The network is trained end-to-end using **Mean Absolute Error (MAE)** loss.

---

## 2. Feature-Based Classification

To classify real images as “used” or “unused,” we extract:

- **Latent features** from the encoder (512-dim)
- **Handcrafted features** including:
  - First-order radiomic statistics
  - GLCM (Gray Level Co-occurrence Matrix) texture features
  - Wavelet subband energies
  - Gabor filter responses
  - Morphological descriptors

These features are concatenated into a comprehensive feature vector. A **Random Forest classifier** is trained on labeled data (100 used, 100 unused real images). Additional anomaly signals are incorporated using **Mahalanobis distances** between real image features and synthetic image distributions.

---

## 3. Mathematical Formulations

### GLCM Contrast
\[
\text{Contrast} = \sum_{i=0}^{N_g -1} \sum_{j=0}^{N_g -1} (i - j)^2 P(i, j)
\]

### Wavelet Subband Energy
\[
E_s = \sum_{m=1}^{M} \sum_{n=1}^{N} |W_s(m, n)|^2
\]

### Gabor Filter Response
\[
G(x, y) = \exp\left(-\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2}\right) \cdot \cos\left(2\pi \frac{x'}{\lambda} + \phi\right)
\]
\[
x' = x \cos\theta + y \sin\theta,\quad y' = -x \sin\theta + y \cos\theta
\]

### Mahalanobis Distance
\[
D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}
\]

---

## 4. Experimental Setup

- **Framework:** TensorFlow 2.12
- **Language:** Python 3.8
- **Training Details:**
  - Epochs: 50
  - Batch size: 32
  - Optimizer: Adam (β₁ = 0.9, β₂ = 0.999)
  - Learning rate: \(1 \times 10^{-4}\)
  - Loss: MAE (Mean Absolute Error)
  - Image resolution: 128 × 128 (RGB)

Training converged within 40–45 epochs. GPU acceleration significantly improved training speed and supported the processing of high-dimensional image data.


