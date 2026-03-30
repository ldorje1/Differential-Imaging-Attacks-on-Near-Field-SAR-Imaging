#### Differential Imaging Attack (DIA) on Classical Millimeter-wave Imaging Algorithms

Run for the codes in this folder to implement DIA on classical algorithms such as MFA, RMA, BPA, LIA (for irregular or non-uniform sensing apertures). 
<!--
Our Differential Imaging Attack is applied to both classical and modern image reconstruction pipelines. Classical algorithms, including the Back-Projection Algorithm (**BPA**), the Range Migration Algorithm (**RMA**), and the Matched Filter Algorithm (**MFA**), can be used to reconstruct the millemeterwave images based directly on the time-domain echo data measurements. The original implementation of these algorithms is not differentiable, can not be used to obtain gradients. We re-implement them as differentiable algorithms in MATLAB based on dlarray. Automatic differentiation engine can then be exploited to calculate gradients, and gradient descent method is the used for realizing DIA. We evaluate their robustness under DIA adversarial attacks.

We also include a more recently imaging algorithm, i.e., [Lightweight Imaging Algorithm (**LIA**)](https://ieeexplore.ieee.org/abstract/document/9362213), an iterative imaging algorithm designed for efficient, high-quality imaging under irregular or non-uniform sensing apertures. One thing to note is that the automatic differentiation through the full LIA operation was very slow because of its large number of iterations. Therefore, for LIA we compute gradients using a linear BPA surrogate built from the same propagation matrix H, while the final attacked image is always reconstructed using the full LIA algorithm.

***

### Attack Implementation
Download the 3D raw data cube 'rawSAR.mat' from the following Google Drive folder: 👉 [Google Drive data folder ](https://drive.google.com/drive/folders/1gymInr98iKLn37k7IIvvssIoM6Zd3r5P?usp=drive_link).
 
For the full DIA attack implementation, the following dataset files (available in the [data folder](https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/tree/main/Traditional%20SAR%20Imaging%20Algos/data)) must be placed in the same directory as the main MATLAB attack script, [DIA_traditional_LIA_main](https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Traditional%20SAR%20Imaging%20Algos/DIA_traditional_LIA_main.m). 

Then, inside the script, select the desired image reconstruction algorithm by setting
sar_algo = 'RMA'; % MFA | RMA | BPA | LIA
| File Name                                 | Description |
|-------------------------------------------|-------------|
| `iqData_noAtk.mat`                        | Clean IQ measurements (Nsamp × nRX × nFrame) |
| `iqData_Atk.mat`                          | Attacked IQ measurements |
| `rawSAR.mat`                              | Raw SAR data cube (`adcDataCube`) (Nsamp x X_axis x Y_axis)|
| `trueImage_complex_MFA.mat`               | Clean MFA complex image |
| `trueImage_complex_RMA.mat`               | Clean RMA complex image |
| `trueImage_complex_BPA.mat`               | Clean BPA complex image |
| `trueImage_complex_LIA.mat`               | Clean LIA complex image |
| `desired_attacked_complex_MFA_RMA.mat`    | Desired target image for MFA/RMA spoofing attack |
| `desired_attacked_complex_BPA_LIA.mat`    | Desired target image for BPA/LIA spoofing attack |

> Clean and attacked complex images are generated in MATLAB using the files in this [folder](https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/tree/main/Traditional%20SAR%20Imaging%20Algos/extra).

> ⚠️ **Important:**  
> The *same* desired target (camouflage) image is used for all algorithms — MFA, RMA, BPA, and LIA.  
> Only the clean reconstruction differs across algorithms.

***

### Clean Results (without DIA)
The figure below shows the baseline reconstructions produced by the traditional SAR algorithms (MFA, RMA, BPA) and the advanced LIA method before applying any DIA. These clean outputs serve as ground-truth references for evaluating how the attack alters different reconstruction pipelines.

<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Traditional%20SAR%20Imaging%20Algos/files/MFA_RMA_BPA_LIA_clean.png"
     width="800" height="800">

⚠️ BPA and LIA images appear blocky because they were generated at 50×50 resolution to reduce the computational cost of forming the H matrix, whereas MFA and RMA images were reconstructed at 401×401 resolution.

***
### Atttack Results 
The figure below shows the final adversarially manipulated SAR reconstructions obtained using our Differential Imaging Attack (DIA) across all four imaging algorithms—MFA, RMA, BPA, and LIA. The DIA optimization uses algorithm-specific learning rates (300 iterations for MFA/RMA/LIA, and 10 iterations for BPA) with lr_re, lr_im ∈ {1e2, 1e3}, an L2 penalty of 1e-4, and a hard magnitude cap of A_max = 2.
 
<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Traditional%20SAR%20Imaging%20Algos/files/MFA_RMA_BPA_LIA_attacked.png"
     width="800" height="800">

-->
