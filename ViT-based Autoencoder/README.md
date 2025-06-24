# ViT-based Autoencoder
Autoencoders are neural networks trained to reconstruct their input. Instead of traditional CNNs, this project uses a Vision Transformer (ViT) as the encoder, showcasing the power of self-attention in unsupervised learning tasks.

**This approach for the ImageCLEF MedicalGAN 2025 - Subtask 1, yielded the best results in terms of accuracy and Kohen-Kappa score**

Key components:
- ViT (Vision Transformer) Encoder 
- Fully-connected / CNN Decoder
- Training pipeline with metrics like accuracy and F1-score
- Visualizations of input vs. reconstructed outputs

## Model Architecture

- **Encoder**:
  - Based on the **ViT-B/16** architecture
  - Modified by:
    - Removing the classification head
    - Adding a projection layer to compress 768-dim token embeddings into a lower-dimensional latent space
    - Using a 1×1 convolution to adapt grayscale (1-channel) images to 3-channel inputs required by ViT
  - Pretrained on a lung CT dataset to improve domain adaptation for medical images. The dataset is available at : [Chest CT Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

- **Decoder**:
  - Convolutional architecture with transposed convolutions
  - Includes residual blocks for better feature propagation and training stability
  - Final Tanh activation to normalize the reconstructed image output
 
The proposed ViT-based autoencoder excels at medical image reconstruction by leveraging self-attention mechanisms to capture global context from the input. Unlike CNNs, which have limited receptive fields, Vision Transformers (ViT) can model long-range dependencies—critical for detecting subtle and spatially dispersed anomalies in medical scans. Additionally, pretraining the encoder on lung CT images enables the model to learn domain-specific anatomical patterns, enhancing its ability to extract meaningful features for reconstruction.

To adapt grayscale medical images to the ViT architecture, a lightweight 1×1 convolution is used for channel expansion. The decoder, built with transposed convolutions and residual blocks, ensures smooth upsampling and stable training, preserving structural integrity in the reconstructed images. A compact latent space bridges the encoder and decoder, effectively capturing semantic content while minimizing information loss, resulting in high-fidelity reconstructions essential for medical analysis.

## Repository Structure
├── ViT_based_autoencoder_.ipynb # Main Jupyter notebook

├── requirements.txt # List of dependencies

├── README.md # Project documentation

Use the following command to install the requirements:

pip install -r requirements.txt


