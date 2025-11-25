Our proposed differential imaging attack (DIA) is applied to three models (CV-Deep2S, Deep2S, Deep2S+) introduced in the paper: "[Efficient Physics-Based Learned Reconstruction Methods for Real-Time 3D Near-Field MIMO Radar Imaging](https://www.sciencedirect.com/science/article/abs/pii/S105120042300369X)".

***
### Three 3D U-Net-based Models from the original paper 
#### Deep2S 
Deep2S is a two-stage, physics-guided learned reconstruction network for 3D near-field MIMO radar imaging.
- Stage 1: Applies the adjoint operator A^H to map raw complex measurements into an intermediate 3D image volume.

- Stage 2: A 3D U-Net refines this intermediate image to suppress artifacts from sparse frequency and antenna sampling.

#### CV-Deep2S (Complex-Valued Deep2S)

CV-Deep2S is a variant of Deep2S that processes the intermediate reconstruction in complex-valued form (real + imaginary channels) instead of magnitude.
- Uses complex-valued layers to refine both real and imaginary parts.

- Achieves higher PSNR than Deep2S but much lower SSIM, and generates more artifacts (especially along the z-axis).

#### Deep2S+
Deep2S+ is an enhanced, fully trainable hybrid version of Deep2S.
Key upgrades:

- Replaces the fixed adjoint operator A^H with a trainable complex-valued projection layer initialized using A^H (physics-based warm start).

- Still uses the 3D U-Net from Deep2S in stage 2, but further fine-tunes both stages jointly.

- Produces higher PSNR/SSIM than Deep2S and reduces z-axis under-sampling artifacts, though sometimes slightly over-smooths.


<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Efficient%20Physics-based%20Learned%20Reconstruction/images/efficient_physics_clean.png"
     width="400" height="400">




***


***

###
