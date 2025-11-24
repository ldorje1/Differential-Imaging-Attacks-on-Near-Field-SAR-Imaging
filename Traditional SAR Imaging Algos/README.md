### Differential Imaging Attack (DIA) on Traditional and Advanced SAR Imaging Algorithms

Our Differential Imaging Attack is applied to both classical and modern SAR image reconstruction pipelines. Traditional algorithms, including the Matched-Filter Algorithm (MFA), Range Migration Algorithm (RMA), and Back-Projection Algorithm (BPA), operate directly on time-domain or frequency-domain SAR measurements and serve as baseline reconstruction models for evaluating DIA performance.

For advanced reconstruction, we include the [Lightweight Imaging Algorithm (LIA)](https://ieeexplore.ieee.org/abstract/document/9362213), an iterative, matrix-based method designed for efficient, high-quality imaging under irregular or non-uniform aperture trajectories.

⚠️⚠️ *One thing to note is that backpropagating through the full LIA operator was very slow because of its iterative nature. Therefore, for LIA we compute gradients using a linear BPA surrogate built from the same propagation matrix H, while the final attacked image is always reconstructed using the full LIA algorithm. I do not know if a trick like this is acceptable in terms of calculating gradients and attacking.*

***

### Attack Implementation for Reproducibility
Please download the 3D raw data cube rawSAR.mat from the following Google Drive folder: 👉 [Google Drive data folder ](https://drive.google.com/drive/folders/1gymInr98iKLn37k7IIvvssIoM6Zd3r5P?usp=drive_link).
 
For the full DIA attack implementation, the following dataset files (available in the [data folder](https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/tree/main/Traditional%20SAR%20Imaging%20Algos/data)) must be placed in the same directory as the main MATLAB attack script, [DIA_traditional_LIA_main](https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Traditional%20SAR%20Imaging%20Algos/DIA_traditional_LIA_main.m). 

Then, inside the script, select the desired SAR reconstruction algorithm by setting
sar_algo = 'RMA'; % MFA | RMA | BPA | LIA
| File Name                                 | Description |
|-------------------------------------------|-------------|
| `iqData_noAtk.mat`                        | Clean IQ measurements (Nsamp × nRX × nFrame) |
| `iqData_Atk.mat`                          | Attacked IQ measurements |
| `rawSAR.mat`                              | Raw SAR data cube (`adcDataCube`) |
| `trueImage_complex_MFA.mat`               | Clean MFA complex image |
| `trueImage_complex_RMA.mat`               | Clean RMA complex image |
| `trueImage_complex_BPA.mat`               | Clean BPA complex image |
| `trueImage_complex_LIA.mat`               | Clean LIA complex image |
| `desired_attacked_complex_MFA_RMA.mat`    | Desired target image for MFA/RMA spoofing attack |
| `desired_attacked_complex_BPA_LIA.mat`    | Desired target image for BPA/LIA spoofing attack |

> ⚠️ **Important:**  
> The *same* desired target (camouflage) image is used for all algorithms — MFA, RMA, BPA, and LIA.  
> Only the clean reconstruction differs across algorithms.

***

### Clean Results (without DIA)
The figure below shows the baseline reconstructions produced by the traditional SAR algorithms (MFA, RMA, BPA) and the advanced LIA method before applying any DIA. These clean outputs serve as ground-truth references for evaluating how the attack alters different reconstruction pipelines.

<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Traditional%20SAR%20Imaging%20Algos/files/MFA_RMA_BPA_LIA_clean.png"
     width="800" height="800">


***
### Atttack Results 
The figure below shows the final adversarially manipulated SAR reconstructions obtained using our Differential Imaging Attack (DIA) across all four imaging algorithms—MFA, RMA, BPA, and LIA. The DIA optimization uses algorithm-specific learning rates (300 iterations for MFA/RMA/LIA, and 10 iterations for BPA) with lr_re, lr_im ∈ {1e2, 1e3}, an L2 penalty of 1e-4, and a hard magnitude cap of A_max = 2.
 
<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Traditional%20SAR%20Imaging%20Algos/files/MFA_RMA_BPA_LIA_attacked.png"
     width="800" height="800">


