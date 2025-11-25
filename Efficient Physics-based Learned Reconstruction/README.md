Our proposed differential imaging attack (DIA) is applied to three models (CV-Deep2S, Deep2S, Deep2S+) introduced in the paper: "[Efficient Physics-Based Learned Reconstruction Methods for Real-Time 3D Near-Field MIMO Radar Imaging](https://www.sciencedirect.com/science/article/abs/pii/S105120042300369X)".
***
### Files Required for the Attack Implementation

**Note:**

1. The experimental measurement data `y_exp_test_4.npy` is first preprocessed according to the requirements of the three models. The preprocessing follows the procedure described in the original paper’s GitHub repository.

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

### UPDATES

### Update: Improved Optimization of Complex Gains for the Deep2S Attack

For the Deep2S attack, we updated how the complex gains **Aₚ** are optimized to make the attack more stable and better behaved under the amplitude constraint **|Aₚ| ≤ A_max**.

---

#### **Old Version (Baseline Approach)**

The original implementation optimized the real and imaginary parts of the gains directly:

A_re ← A_re − η_re * ∇{A_re} L
A_im ← A_im − η_im * ∇{A_im} L


After each update, the amplitude constraint was enforced via hard projection:

1. Compute |Aₚ| for each aperture.
2. If |Aₚ| > A_max, rescale that entry so that |Aₚ| = A_max.

This approach caused:

- Many entries to sit *exactly* at the boundary |Aₚ| = A_max  
- Oscillatory or unstable loss behavior  
- Sensitivity to learning-rate choice and abrupt clipping  

---

#### **New Version (Current Implementation)**

We now optimize unconstrained latent variables **Z_re**, **Z_im**, and map them smoothly into **A**:

A_re = (A_max / 2) * tanh(Z_re)
A_im = (A_max / 2) * tanh(Z_im)


This guarantees **|Aₚ| ≤ A_max** automatically, without explicit clipping, and provides smoother gradients near the constraint boundary.

Key improvements:

- **Adam optimizer** is used on *(Z_re, Z_im)* for stable adaptive updates.  
- **Amplitude bound is satisfied by construction** through the tanh mapping.  
- An L2-regularization term keeps the optimizer from trivially pushing all amplitudes toward A_max.  
- The attack converges more smoothly and uses power more coherently across apertures.

---

#### **Summary of the Change**

We replaced the old:

> “optimize A directly + hard clipping”

with a new:

> “optimize latent variables Z + smooth tanh constraint + Adam”

under the same physical amplitude boundary **|Aₚ| ≤ A_max**.

This results in **more stable optimization, fewer saturations, better gradient behavior**, and ultimately **more effective Deep2S attacks**.



