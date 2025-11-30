# Differential Imaging Attacks on Near-Field SAR Imaging
***
### (Applied to RMIST-Net)
Our proposed Differential Imaging Attack (DIA) is applied to the range migration (RM) kernel-based iterative-shrinkage thresholding network 9RMIST-Net), introduced in the paper: [RMIST-Net: Joint Range Migration and Sparse Reconstruction Network for 3-D mmW Imaging](https://ieeexplore.ieee.org/document/9393590). 

RMIST-Net is a physics-guided unrolled sparse imaging network for mmWave/SAR data. It replaces the large CS sensing matrix with a fast FFT/IFFT range-migration operator and unrolls ISTA-style iterations into T learnable phases.

(14)  S = IFT2D( FT2D(alpha) * Phi_r )  = RM(alpha)

(15)  alpha = IFT2D( FT2D(S) * Phi_r_conj )  = RM_dagger(S)



**📌 Note:**

(1) Because the original RMIST-NET model does not provide pretrained weights, we reconstructed the unrolled network from the paper and re-trained it using a synthetic point-target dataset generated in MATLAB ([link](https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/RMIST-Net/extra/training_data_synth_main.m)).   



## 📁 Files Required for the Attack Implementation
