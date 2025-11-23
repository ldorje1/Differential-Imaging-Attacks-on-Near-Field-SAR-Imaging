# Differential Imaging Attacks on Near-Field SAR Imaging

This repository contains the official implementation of  
**“Differential Imaging Attacks on Near-Field SAR Imaging.”**

The project provides a complete pipeline for constructing and optimizing adversarial perturbations targeting classical and modern SAR imaging models. The attack operates directly in the RF measurement domain and optimizes image-level objectives while respecting hardware and signal constraints.

---

## 🚀 Features

- **Signal Alignment Framework**
  - Range-bin alignment
  - Complex gain correction
  - Gate-based range selection

- **Attack Signal Extraction**
  - Clean vs. attacked measurement comparison  
  - Per-look correction via time-shifts and complex scalars  
  - Construction of the attack pool \(X_{aa}\)

- **Forward Modeling**
  - Classical SAR forward models  
  - Frequency-domain and time-domain operators  
  - Complex-valued autodiff support (PyTorch / MATLAB)

- **Adversarial Optimization**
  - Image-domain losses  
  - Power-constrained amplitude/phase optimization  
  - Wirtinger-gradient computation  
  - Projected gradient descent (PGD)

- **Supported Imaging Algorithms**
  - Back-Projection Algorithm (BPA)  
  - Matched Filter Algorithm (MFA)  
  - Range Migration Algorithm (RMA)  
  - Compressed Sensing Algorithm (CSA)  
  - Lightweight Imaging Algorithm (LIA)  
  - Unrolled SAR networks  
  - Deep learning SAR reconstructor models

---

## 📦 Repository Structure
differential-sar-attack/
│
├── alignment/ # Range/gate alignment, α–τ estimation
├── attack_pool/ # Construction of X_aa from r0, r1
├── forward_models/ # MFA, RMA, BPA, CSA, LIA, etc.
├── optimization/ # Image-domain adversarial solver
├── models/ # Unrolled + DNN SAR reconstructors
├── utils/ # Helpers for plotting, metrics, etc.
└── examples/ # End-to-end attack demos
