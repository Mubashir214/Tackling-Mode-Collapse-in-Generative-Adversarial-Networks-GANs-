# GAN Mode Collapse Solution using DCGAN & WGAN-GP

## README.md

# 🎨 Tackling Mode Collapse in GANs using DCGAN & WGAN-GP

This project demonstrates the implementation and comparison of two popular Generative Adversarial Network (GAN) architectures:

* **DCGAN (Deep Convolutional GAN)** — Baseline GAN model
* **WGAN-GP (Wasserstein GAN with Gradient Penalty)** — Improved GAN for stable training and reduced mode collapse

The main objective is to analyze how advanced loss functions and training strategies improve image diversity and training stability in GANs.

---

# 🚀 Live Demo

### Streamlit Application

[Live Demo – GAN Comparison App](https://8tplcjekvnmlxx7j8sjckc.streamlit.app/?utm_source=chatgpt.com)

The app allows users to:

* Generate images using both models
* Compare DCGAN vs WGAN-GP outputs
* Observe differences in diversity and quality
* Perform real-time image generation

---

# 📌 Project Objectives

This project aims to:

* Build a baseline **DCGAN**
* Implement advanced **WGAN-GP**
* Address the issue of **mode collapse**
* Improve training stability
* Compare image diversity between both models
* Deploy the system using Streamlit

---

# 🧠 Concepts Covered

* Generative Adversarial Networks (GANs)
* Deep Convolutional GANs (DCGAN)
* Wasserstein Loss
* Gradient Penalty
* Mixed Precision Training
* Stable GAN Training
* Image Generation
* Mode Collapse Reduction

---

# 📂 Dataset Used

## 1️⃣ Pokemon Sprites Dataset

[Pokemon Sprites Dataset](https://www.kaggle.com/datasets/jackemartin/pokemon-sprites?utm_source=chatgpt.com)

## 2️⃣ Anime Faces Dataset (64×64)

[Anime Faces Dataset](https://www.kaggle.com/datasets/soumikrakshit/anime-faces?utm_source=chatgpt.com)

---

# ⚙️ Environment Setup

## Platform

* Kaggle Notebook

## Hardware

* GPU: Tesla T4 ×2

## Libraries Used

```bash
torch
torchvision
numpy
matplotlib
streamlit
Pillow
tqdm
```

Install dependencies:

```bash
pip install torch torchvision matplotlib streamlit tqdm pillow
```

---

# 🏗️ Model Architectures

# 1️⃣ DCGAN (Baseline Model)

## Generator

* Input Noise Vector: 100-dimensional
* Transposed Convolution Layers
* Batch Normalization
* ReLU Activation
* Output Activation: Tanh

## Discriminator

* Convolution Layers
* LeakyReLU Activation
* Output Activation: Sigmoid

## Loss Function

* Binary Cross Entropy Loss (BCE)

---

# 2️⃣ WGAN-GP (Advanced Model)

## Improvements over DCGAN

* Replaces Discriminator with Critic
* Removes Sigmoid Activation
* Uses Wasserstein Loss
* Applies Gradient Penalty
* Stable training dynamics
* Reduces mode collapse

## Key Hyperparameters

| Parameter          | Value        |
| ------------------ | ------------ |
| Learning Rate      | 0.0002       |
| Betas              | (0.5, 0.999) |
| Gradient Penalty λ | 10           |
| Critic Updates     | 5            |
| Batch Size         | 64           |

---

# 📁 Project Structure

```bash
GAN-Mode-Collapse/
│
├── notebooks/
│   ├── dcgan_training.ipynb
│   ├── wgan_gp_training.ipynb
│
├── models/
│   ├── generator.py
│   ├── discriminator.py
│   ├── critic.py
│
├── checkpoints/
│   ├── dcgan/
│   ├── wgan_gp/
│
├── outputs/
│   ├── generated_images/
│   ├── training_plots/
│
├── app/
│   ├── streamlit_app.py
│
├── requirements.txt
├── README.md
```

---

# 📊 Data Preparation

The preprocessing pipeline includes:

1. Loading dataset
2. Resizing images to 64×64
3. Converting images to tensors
4. Normalizing images to range [-1, 1]
5. Creating PyTorch DataLoader

Example normalization:

```python
transforms.Normalize((0.5,), (0.5,))
```

---

# 🔄 Training Process

## DCGAN Training

Steps:

1. Generate random noise vector
2. Produce fake image
3. Pass real & fake images through discriminator
4. Compute BCE loss
5. Update Generator & Discriminator

---

## WGAN-GP Training

Steps:

1. Generate fake images
2. Compute critic scores
3. Calculate Wasserstein loss
4. Apply Gradient Penalty
5. Update critic multiple times
6. Update generator

---

# ⚡ Optimization Techniques

To fit training on Kaggle T4 GPUs:

* Mixed Precision Training (`torch.cuda.amp`)
* Batch Size Optimization
* Dataset Subsetting
* Checkpoint Saving every 5–10 epochs
* GPU Memory Monitoring

---

# 📈 Training Logs

The project includes:

* Generator Loss vs Epochs
* Discriminator/Critic Loss vs Epochs
* Generated Image Samples

Visualization examples:

* DCGAN generated outputs
* WGAN-GP generated outputs
* Side-by-side comparison

---

# 🖼️ Results & Comparison

| Feature            | DCGAN    | WGAN-GP     |
| ------------------ | -------- | ----------- |
| Training Stability | Medium   | High        |
| Mode Collapse      | Common   | Rare        |
| Image Diversity    | Lower    | Higher      |
| Convergence        | Unstable | Stable      |
| Loss Function      | BCE      | Wasserstein |

---

# 🔍 Observations

## DCGAN

* Generates decent images
* Often suffers from mode collapse
* Training becomes unstable after several epochs

## WGAN-GP

* Produces more diverse images
* Stable learning behavior
* Better quality outputs
* Improved convergence

---

# 📱 Streamlit Deployment

Run locally:

```bash
streamlit run streamlit_app.py
```

Live deployed application:

[Open Streamlit GAN App](https://8tplcjekvnmlxx7j8sjckc.streamlit.app/?utm_source=chatgpt.com)

---

# 📌 Future Improvements

* Add FID Score Evaluation
* Add Inception Score (IS)
* Train on larger anime datasets
* Conditional GAN implementation
* StyleGAN integration
* Multi-GPU Distributed Training

---

# 🎯 Conclusion

This project successfully demonstrates how advanced GAN techniques like **WGAN-GP** significantly improve GAN training stability and reduce mode collapse compared to traditional **DCGAN** models.

The comparison clearly shows:

* Better diversity
* Stable optimization
* Improved image realism

WGAN-GP proves to be a more robust solution for generative image modeling tasks.

---

# 👨‍💻 Author

**Mubashir Siddique**

AI / Deep Learning / Computer Vision Enthusiast

---

# 📜 License

This project is developed for educational and research purposes.
