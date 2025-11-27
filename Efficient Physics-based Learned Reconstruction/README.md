# Differential Imaging Attacks on Near-Field SAR Imaging
### (Applied to Efficient Physics-Based 3D Learned Reconstruction Models)

Our proposed Differential Imaging Attack (DIA) is applied to the three models (CV-Deep2S, Deep2S, Deep2S+) introduced in the paper: "[Efficient Physics-Based Learned Reconstruction Methods for Real-Time 3D Near-Field MIMO Radar Imaging](https://www.sciencedirect.com/science/article/abs/pii/S105120042300369X)".

#### Three 3D U-Net Models From the Paper (Reference)
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

Example Images from the Original Paper (re-arranged for clarity).

<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Efficient%20Physics-based%20Learned%20Reconstruction/images/efficient_physics_clean.png"
     width="400" height="400">
     


***
### Files Required for the Attack Implementation
Use DIA_Deep2S_main.py, DIA_CVDeep2S_main.py, or DIA_Deep2SP_main.py to run the attack. All required files for the corresponding models can be downloaded from:👉 **[Google Drive](https://drive.google.com/drive/u/1/folders/1gymInr98iKLn37k7IIvvssIoM6Zd3r5P)**.


Place all of the following files in the working directory before running any DIA script:
| File | Description |
|------|-------------|
| **A15_exp.npy** | Precomputed propagation matrix **A** from the original Deep2S paper. Required for all three models. |
| **D_flat.npy** | Our preprocessed experimental attack dictionary, generated in MATLAB to match the model requirements. |
| **y_exp_test_4.npy** | Raw experimental measurement used for testing (from the original paper). |
| **model_Nf15_SNR30_exp.h5** | The pretrained Deep2S model weights provided by the authors. |
| **model_Nf15_SNR30_CV_exp.h5** | The pretrained CVDeep2S model weights provided by the authors. |
| **model_Nf15_SNR30_Deep2SP_exp** | The pretrained Deep2S+ model weights provided by the authors. |
| **src.py** | Main inference/attack routines (method definitions for preprocessing, projection layer, and forward pass). |
| **misc.py** | Utility functions used by the model (normalization, FFT helpers, padding, etc.). |

**📌 Note:**
0. All three model use same files except the trained weights.

1. `y_exp_test_4.npy` is first preprocessed in MATLAB according to the Deep2S / CV-Deep2S / Deep2S+ pipeline. This follows the procedure described in the original authors’ GitHub.

2. We use the original experimental measurement (not our own) because the propagation matrix A was computed from the original measurement environment, and is not compatible with other systems.

3. Our attack dictionary `D_flat.npy` was computed and preprocessed in MATLAB to meet the model’s dimensional and normalization requirements.

***
# (1) DIA on Deep2S 
We implemented two variants of the differential imaging attack, depending on where the loss is computed:

🛠️ **(1) Global Loss (Full-Image Attack):**  :The L2 loss is computed over all pixels in the reconstructed image. This forces the attack to globally reshape the entire SAR volume toward the target. Stronger but more power-demanding; produces large structural changes.

L_global = || I_attacked  –  I_target ||_2   (all pixels)

```text
Iter   1/300 | Loss=1.754532e+00 | loss_im=1.754532e+00 | reg=3.843202e-11 | mean|A|=1.966e-01, max|A|=2.013e-01
Iter   5/300 | Loss=1.707187e+00 | loss_im=1.707185e+00 | reg=2.148420e-06 | mean|A|=4.936e-01, max|A|=8.887e-01
Iter  10/300 | Loss=1.678663e+00 | loss_im=1.678657e+00 | reg=6.122017e-06 | mean|A|=7.601e-01, max|A|=1.444e+00

Iter 290/300 | Loss=1.285002e+00 | loss_im=1.284976e+00 | reg=2.644721e-05 | mean|A|=1.579e+00, max|A|=1.995e+00
Iter 295/300 | Loss=1.282835e+00 | loss_im=1.282809e+00 | reg=2.649321e-05 | mean|A|=1.582e+00, max|A|=1.995e+00
Iter 300/300 | Loss=1.281767e+00 | loss_im=1.281740e+00 | reg=2.658635e-05 | mean|A|=1.585e+00, max|A|=1.995e+00
Attack optimization finished.
```
```text
Amax: 2.0  
lambda_L2: 1e-5  
max_iter: 300  
initial_learning_rate: 1e-1  
low_learning_rate: 1e-2  
lr_switch_iter: 100  

```
<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Efficient%20Physics-based%20Learned%20Reconstruction/images/full_image_attacked_deep2s.png"
     width="600" height="600">
```text
MSE: 1.7197e+00
RMSE: 1.3114e+00
MAE: 1.0403e+00
NCC: 0.9195
SSIM: 0.0119
PSNR: 8.65 dB
```



🛠️ **(2) ROI Loss (Object-Focused Attack):**: The L2 loss is computed only inside a predefined Region of Interest (ROI). The ROI corresponds to where the main object or target is located. This yields: Lower required perturbation power, more localized edits and minimal distortion outside the object region. 

L_ROI = || I_attacked[ROI]  –  I_target[ROI] ||_2

<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Efficient%20Physics-based%20Learned%20Reconstruction/images/roi_image.png"
     width="600" height="600">
     
```text
Iter   1/300 | Loss=4.373837e+00 | loss_im=4.373837e+00 | reg=4.189320e-11 | mean|A|=1.989e-01, max|A|=2.036e-01
Iter   5/300 | Loss=4.139207e+00 | loss_im=4.139205e+00 | reg=2.231845e-06 | mean|A|=5.030e-01, max|A|=8.978e-01
Iter  10/300 | Loss=3.901488e+00 | loss_im=3.901482e+00 | reg=5.846005e-06 | mean|A|=7.450e-01, max|A|=1.415e+00

Iter 295/300 | Loss=7.075801e-01 | loss_im=7.075483e-01 | reg=3.175850e-05 | mean|A|=1.764e+00, max|A|=1.985e+00
Iter 300/300 | Loss=7.071160e-01 | loss_im=7.070842e-01 | reg=3.176873e-05 | mean|A|=1.764e+00, max|A|=1.985e+00
Attack optimization finished.
```

```text
Amax: 2.0  
lambda_L2: 1e-5  
max_iter: 300  
Z_init_scale: 1e-3  
initial_learning_rate: 1e-1  
lr_switch_iter: 100  
low_learning_rate: 5e-2  
roi_weight: 1.0  
bg_weight: 0.0  
```

<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Efficient%20Physics-based%20Learned%20Reconstruction/images/roi_attacked_deep2s.png"
     width="600" height="600">

```text
MSE: 2.1862e+01
RMSE: 4.6757e+00
MAE: 2.4364e+00
NCC: 0.3439
SSIM: 0.0060
PSNR: -2.39 dB

Clean vs Target (ROI): 4.3739e+00
Attacked vs Target (ROI): 7.0700e-01
```
***
# (2) DIA on CV-Deep2S
🛠️ **(1) Global Loss (Full-Image Attack) Result:** Clean image, desired target, and DIA attacked image

<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Efficient%20Physics-based%20Learned%20Reconstruction/images/full_image_attacked_cvdeep2s.png"
     width="600" height="600">

🛠️ **(2) ROI Loss (Object-Focused Attack):** Clean image, ROI mask, and mask ovelayed on clean image

<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Efficient%20Physics-based%20Learned%20Reconstruction/images/roi_image_CVdeep2s.png"
     width="600" height="600">

**Result**

<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Efficient%20Physics-based%20Learned%20Reconstruction/images/roi_attacked_CVdeep2s.png"
     width="600" height="600">

***

# (3) DIA on Deep2S+
🛠️ **(1) Global Loss (Full-Image Attack) Result:** Clean image, desired target, and DIA attacked image

<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Efficient%20Physics-based%20Learned%20Reconstruction/images/full_image_attacked_deep2s%2B.png"
     width="600" height="600">"

🛠️ **(2) ROI Loss (Object-Focused Attack):** Clean image, ROI mask, and mask ovelayed on clean image

<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Efficient%20Physics-based%20Learned%20Reconstruction/images/roi_image_deep2s%2B.png"
     width="600" height="600">"

**Result**

<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Efficient%20Physics-based%20Learned%20Reconstruction/images/roi_attacked_deep2s%2B.png"
     width="600" height="600">"



***

### 🔧 [Some Changes: Attack Optimization Update (Compared to DIA on tradiational and LIA algo)](https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Efficient%20Physics-based%20Learned%20Reconstruction/files/implementation_update.pdf)

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





