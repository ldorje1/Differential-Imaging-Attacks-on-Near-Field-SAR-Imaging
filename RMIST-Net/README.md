# Differential Imaging Attacks on Near-Field SAR Imaging
***
### (Applied to RMIST-Net)
Our proposed Differential Imaging Attack (DIA) is applied to the range migration (RM) kernel-based iterative-shrinkage thresholding network 9RMIST-Net), introduced in the paper: [RMIST-Net: Joint Range Migration and Sparse Reconstruction Network for 3-D mmW Imaging](https://ieeexplore.ieee.org/document/9393590). 

RMIST-Net is a physics-guided unrolled sparse imaging network for mmWave/SAR data. It replaces the large CS sensing matrix with a fast FFT/IFFT range-migration operator and unrolls ISTA-style iterations into T learnable phases.


### Range Migration (RM) Operator

The forward RM operator is defined as:

\[
\mathbf{S} = \text{IFT}_{2D}\!\left( \text{FT}_{2D}(\boldsymbol{\alpha}) \odot \Phi_r \right)
\;\triangleq\; \text{RM}(\boldsymbol{\alpha})
\tag{14}
\]

and the adjoint (back-projection) operator is:

\[
\boldsymbol{\alpha} = \text{IFT}_{2D}\!\left( \text{FT}_{2D}(\mathbf{S}) \odot \Phi_r^\dagger \right)
\;\triangleq\; \text{RM}^{\dagger}(\mathbf{S})
\tag{15}
\]


**📌 Note:**

(1) Because the original RMIST-NET model does not provide pretrained weights, we reconstructed the unrolled network from the paper and re-trained it using a synthetic point-target dataset generated in MATLAB ([link](https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/RMIST-Net/extra/training_data_synth_main.m)).   



## 📁 Files Required for the Attack Implementation
