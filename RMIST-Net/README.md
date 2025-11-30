# Differential Imaging Attacks on Near-Field SAR Imaging
***
### (Applied to RMIST-Net)
Our proposed Differential Imaging Attack (DIA) is applied to the range migration (RM) kernel-based iterative-shrinkage thresholding network 9RMIST-Net), introduced in the paper: [RMIST-Net: Joint Range Migration and Sparse Reconstruction Network for 3-D mmW Imaging](https://ieeexplore.ieee.org/document/9393590). 

RMIST-Net is a physics-guided unrolled sparse imaging network for mmWave/SAR data. It replaces the large CS sensing matrix with a fast FFT/IFFT range-migration operator and unrolls ISTA-style iterations into T learnable phases.

#### Embed RM Kernel Into ISTA
Phi_r      = k_y^(-1) * exp(-j * k_y * r)

Phi_r_dag  = k_y * exp(j * k_y * r)

S = IFT2D( FT2D(alpha) ⊙ Phi_r )  ≜ RM(alpha)

alpha = IFT2D( FT2D(S) ⊙ Phi_r^† )  ≜ RM†(S)

**📌 Note:**

(1) Because the original RMIST-NET model does not provide pretrained weights, we reconstructed the unrolled network from the paper and re-trained ([re-trained weights](https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/tree/main/RMIST-Net/models)) it using a synthetic point-target dataset generated in MATLAB ([link](https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/RMIST-Net/extra/training_data_synth_main.m)).   

***

## 📁 Files Required for the Attack Implementation

Place **all** of the following files in the working directory before running the RMIST attack scripts:

| **File** | **Description** |
|----------|-----------------|
| **rawSAR.mat** | Raw SAR measurement cube. Variable: `adcDataCube` of size **(Nsamp × M × N)**, complex. |
| **D.mat** | Flattened attack dictionary. Variable: `D` of size **(Nsamp × N_p)** (or **Nsamp × M·N**), complex; generated in MATLAB. |
| **trueImage_complex_RMA_RMIST_512.mat** | Ground-truth complex SAR image for RMIST/RMA. Variable: `trueImage_complex` of size **(H × W)**. |
| **desired_attacked_complex_RMA_RMIST.mat** | Desired target (camouflaged) complex image. Variable: `sar_camouflaged` of size **(H × W)**. |
| **global_max.mat** | Global normalization scalar used to match RMIST training scale. Variable: `global_max`. |
| **S_hat_knife.mat** | Precomputed RMIST intermediate output for the *knife* object. Variable: `S_hat`. |

> **Note:** All `.mat` files were generated using MATLAB during RMIST preprocessing and reconstruction.


*** 
## Re-trained RMNIST-Net Clean Results

```text
# Run re-trained model
with torch.no_grad():
    pred_ri_realimag = model(echo_ri_tensor)   # (1, 2, H, W)

pred_ri_realimag = pred_ri_realimag[0].cpu().numpy()   # (2, H, W)

# Visualize reconstructed magnitude
mag_pred      = np.sqrt(pred_ri_realimag[0]**2 + pred_ri_realimag[1]**2)
mag_pred_norm = mag_pred / (mag_pred.max() + 1e-12)

plt.figure(figsize=(5, 5))
plt.title("Reconstructed SAR Image (RMIST-Net)")
plt.imshow(mag_pred_norm, cmap='jet', origin='lower')
plt.colorbar()
plt.axis('off')
plt.show()
```

<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/RMIST-Net/figures/clean_rmnist_net.png"
     width="500" height="500">

## DI-Attack on Re-trained RMNIST-Net

<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/RMIST-Net/figures/attacked_rmnist_net.png"
     width="500" height="500">

