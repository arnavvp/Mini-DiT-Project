# Diffusion Transformer (DiT) for Class Conditional Image Generation

A PyTorch implementation of a **Diffusion Transformer (DiT)** built by extending a custom **Vision Transformer (ViT)** into a **Denoising Diffusion Probabilistic Model (DDPM)** for image generation on the CIFAR-10 dataset.

The project implements the complete diffusion pipeline from scratch, including transformer-based noise prediction, Gaussian forward diffusion, iterative reverse sampling, and conditional image generation using class embeddings.

---

## Features

- Vision Transformer (ViT) implemented from scratch
- Patch Embedding and Positional Encoding
- Multi-Head Self-Attention Transformer Encoder
- Diffusion Transformer (DiT) architecture
- DDPM Forward Diffusion
- Gaussian Noise Scheduler
- Learned Timestep Embeddings
- Transformer-based Noise Prediction
- 1000-Step Reverse Diffusion Sampling
- Class-Conditional Image Generation using Label Embeddings
- CIFAR-10 Training Pipeline
- Denoising Visualization and Evaluation
- GPU Training Support (CUDA)

---

## Architecture

```
                Image
                  │
          Patch Embedding
                  │
        Positional Encoding
                  │
      + Timestep Embedding
                  │
     + Class Label Embedding
                  │
        Transformer Encoder × N
                  │
        Patch-wise Noise Predictor
                  │
          Predicted Gaussian Noise
```

---

## Diffusion Pipeline

### Forward Process

- Sample timestep `t`
- Add Gaussian noise according to the DDPM scheduler
- Generate noisy image

```
Image → Noise Scheduler → Noisy Image
```

---

### Reverse Process

The transformer predicts the added noise at each timestep.

```
Random Noise
      ↓
DiT predicts noise
      ↓
Remove predicted noise
      ↓
Repeat for 1000 steps
      ↓
Generated Image
```

---

## Model Details

| Component | Value |
|----------|------:|
| Dataset | CIFAR-10 |
| Training Images | 50,000 |
| Test Images | 10,000 |
| Image Resolution | 32 × 32 |
| Transformer Layers | 6 |
| Attention Heads | 8 |
| Embedding Dimension | 128 |
| Diffusion Steps | 1000 |
| Optimizer | AdamW |
| Loss | Mean Squared Error (Noise Prediction) |

---

## Evaluation

The model is evaluated using:

- Noise Prediction MSE
- Image Reconstruction MSE
- PSNR
- FID (planned)
- Progressive Denoising Visualization

Example results:

| Metric | Value |
|---------|------:|
| Average Noise Prediction MSE | ~0.13 |
| High-Noise Prediction MSE | <0.03 |
| Low-Noise Reconstruction MSE | ~0.015 |

---


## Training

```bash
python -m src.engine.train
```

---

## Sampling

Generate images from pure Gaussian noise:

```bash
python -m src.sample
```

---

## Denoising Evaluation

Evaluate reconstruction quality:

```bash
python -m src.test_denoise
```

---

## Future Improvements

- Classifier-Free Guidance (CFG)
- Faster DDIM Sampling
- Larger DiT Architectures
- FID Optimization
- Mixed Precision Training
- Higher Resolution Image Generation
- Latent Diffusion Models
- Image-to-Image Diffusion

---

## Tech Stack

- Python
- PyTorch
- CUDA
- Vision Transformers (ViT)
- Diffusion Transformers (DiT)
- Denoising Diffusion Probabilistic Models (DDPM)
- NumPy
- Matplotlib

---

## References

- **An Image is Worth 16x16 Words** (Vision Transformer)
- **Scalable Diffusion Models with Transformers (DiT)**
- **Denoising Diffusion Probabilistic Models (DDPM)**

---

## Author

**Arnav Patel**
