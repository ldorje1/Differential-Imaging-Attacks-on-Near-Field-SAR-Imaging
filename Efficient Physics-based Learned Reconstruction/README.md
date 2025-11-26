Our proposed differential imaging attack (DIA) is applied to three models (CV-Deep2S, Deep2S, Deep2S+) introduced in the paper: "[Efficient Physics-Based Learned Reconstruction Methods for Real-Time 3D Near-Field MIMO Radar Imaging](https://www.sciencedirect.com/science/article/abs/pii/S105120042300369X)".
***
### Files Required for the Attack Implementation
All required files for the corresponding models can be downloaded from:👉 **[Google Drive](https://drive.google.com/drive/u/1/folders/1gymInr98iKLn37k7IIvvssIoM6Zd3r5P)**.


The following files must be placed in the working directory before running the Deep2S / Deep2S+ / CV-Deep2S attack scripts:

| File | Description |
|------|-------------|
| **A15_exp.npy** | Precomputed propagation matrix **A** from the original Deep2S paper. Required for all three models. |
| **D_flat.npy** | Our preprocessed experimental attack dictionary, generated in MATLAB to match the model requirements. |
| **y_exp_test_4.npy** | Raw experimental measurement used for testing (from the original paper). |
| **model_Nf15_SNR30_exp.h5** | The pretrained Deep2S / Deep2S+ model weights provided by the authors. |
| **src.py** | Main inference/attack routines (method definitions for preprocessing, projection layer, and forward pass). |
| **misc.py** | Utility functions used by the model (normalization, FFT helpers, padding, etc.). |

**Note:**

1. The experimental measurement data `y_exp_test_4.npy` is first preprocessed in MATLAB according to the requirements of the three models. The preprocessing follows the procedure described in the original paper’s GitHub repository.

2. We use the raw measurement data (to generated `y_exp_test_4.npy`) provided by the paper (instead of our own) because the predefined propagation matrix **A** used by all models is derived from the original measurement environment.

3. Our baseline experimental attack dictionary `D_flat.npy` was computed and preprocessed in MATLAB to meet the model requirements.





***
### Three 3D U-Net-based Models from the Original Paper (for reference) 
*Deep2S*: Deep2S is a two-stage, physics-guided learned reconstruction network for 3D near-field MIMO radar imaging.
- Stage 1: Applies the adjoint operator A^H to map raw complex measurements into an intermediate 3D image volume.

- Stage 2: A 3D U-Net refines this intermediate image to suppress artifacts from sparse frequency and antenna sampling.

*CV-Deep2S (Complex-Valued Deep2S)*: CV-Deep2S is a variant of Deep2S that processes the intermediate reconstruction in complex-valued form (real + imaginary channels) instead of magnitude.
- Uses complex-valued layers to refine both real and imaginary parts.

- Achieves higher PSNR than Deep2S but much lower SSIM, and generates more artifacts (especially along the z-axis).

*Deep2S+*: Deep2S+ is an enhanced, fully trainable hybrid version of Deep2S. Key upgrades:

- Replaces the fixed adjoint operator A^H with a trainable complex-valued projection layer initialized using A^H (physics-based warm start).

- Still uses the 3D U-Net from Deep2S in stage 2, but further fine-tunes both stages jointly.

- Produces higher PSNR/SSIM than Deep2S and reduces z-axis under-sampling artifacts, though sometimes slightly over-smooths.

The following are generated images from the original paper (re-arranged for clarity).
<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Efficient%20Physics-based%20Learned%20Reconstruction/images/efficient_physics_clean.png"
     width="400" height="400">

***
### DIA on the Models 

***

### 🔧 Summary of Changes: Deep2S Attack Optimization Update

We updated the optimization strategy for the complex gains **Aₚ** in the Deep2S Differential Imaging Attack (DIA) to improve stability and behavior under the physical amplitude constraint **|Aₚ| ≤ A_max**.

#### **Old Method (Baseline)**
- Optimized the real and imaginary parts of A directly using gradient descent.
- Applied **hard clipping** after each update: if |Aₚ| > A_max, rescale back to the boundary.
- Resulted in:
  - Many gains stuck at |Aₚ| = A_max  
  - Unstable or oscillatory loss  
  - Sensitivity to step sizes and abrupt clipping

#### **New Method (Current Implementation)**
- Introduced **latent variables** Z_re and Z_im, optimized *unconstrained*.
- Mapped them to A using a smooth tanh transformation:
  - Ensures **|Aₚ| ≤ A_max** automatically (no clipping needed)
  - Provides smoother gradients near the constraint boundary
- Replaced manual gradient descent with **Adam** for adaptive and stable updates.
- Added an **L2 penalty** on |A| to prevent trivial max-amplitude solutions and encourage structured use of the power budget.
- Backpropagation now differentiates through the tanh mapping using the chain rule.

#### **Outcome**
- Optimization is significantly more stable.
- No more abrupt clipping or saturation at A_max.
- Better gradient flow and smoother convergence.
- The attack produces more consistent and physically realistic perturbations.





