# Dual-Contrastive GAN for Feature Attribution in CT Imaging

This repository implements **DCLGAN (Dual-Contrastive GAN)**, a GAN-based framework designed to extract, analyze, and compare feature attribution patterns from medical CT scan images. The method enables interpretability by examining the discriminator’s spatial attention across three domains: `real_used`, `real_not_used`, and `generated` images.

---

## Objective

The DCLGAN framework is intended to:
- Extract feature maps from discriminator activations
- Quantify spatial attention overlaps between real and synthetic CT slices
- Preserve medically meaningful intensity distributions using a novel **Hounsfield Unit (HU) Loss**
- Support attribution analysis to identify which real images most influenced GAN training

---

## Architecture Overview

### Generator

The generator follows a ResNet-style architecture with downsampling and upsampling:

- **Input**: 256×256 CT image (1 channel)
- $7 \times 7$ Convolution (64 filters)
- Downsampling:
  - $3 \times 3$ Conv (stride 2) → 128 filters
  - $3 \times 3$ Conv (stride 2) → 256 filters
- **Residual Blocks**: 8 blocks at 256 channels with identity shortcuts
- Upsampling:
  - Bilinear interpolation + $3 \times 3$ Conv → 128
  - Bilinear interpolation + $3 \times 3$ Conv → 64
- $7 \times 7$ Conv + Tanh activation to produce final image

### Discriminator (PatchGAN-based)

- 5 Conv2D layers: 64 → 128 → 256 → 512 → 1 channels
- LeakyReLU activations and Instance Normalization
- Final output: Patch-wise real/fake prediction map
- Forward hooks placed on the last 3 post-activation layers to extract **feature maps** for attribution analysis

---

## Experimental Setup

- **Framework**: PyTorch 2.0
- **Hardware**: NVIDIA GPU (CUDA-enabled)
- **Epochs**: 200
- **Batch Size**: 1 (standard for unpaired image translation)
- **Optimizer**: Adam with TTUR
  - Generator: LR = 1e-4, β=(0.5, 0.9)
  - Discriminator: LR = 4e-4, β=(0.5, 0.999)
- **Schedulers**: CosineAnnealingLR for both G and D
- **Regularization**:
  - Spectral normalization on Conv2D/Linear layers in the discriminator
  - Gradient penalty

---

## Loss Functions

### 1. **Hounsfield Unit (HU) Loss** (Novel)
Preserves CT intensity distributions by minimizing KL-divergence between histograms of real and generated images.

\[
\mathcal{L}_{\text{HU}} = \sum_{i=1}^{N} P_r(i) \log \left( \frac{P_r(i) + \epsilon}{P_f(i) + \epsilon} \right)
\]

Where:
- \( P_r(i), P_f(i) \): Histogram probabilities for real and fake
- \( \epsilon \): Small constant for numerical stability

### 2. **PatchNCE Loss**
Localized contrastive loss aligning real and generated patches.

### 3. **Feature Matching Loss**
Stabilizes training by L1-matching discriminator activations from real and fake inputs.

---

## Attribution Hypothesis

The extracted feature maps are analyzed to quantify discriminator attention similarity. We hypothesize that:

> **Feature attention in "generated" images will align more closely with "real_used" images than with "real_not_used".**

This supports interpretability and training-data influence tracing.