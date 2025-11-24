### Traditional and Advanced SAR Imaging Algorithms

Our Differential Imaging Attack (DIA) is applied to both classical and modern SAR image reconstruction pipelines. Traditional algorithms—including the Matched-Filter Algorithm (MFA), Range Migration Algorithm (RMA), and Back-Projection Algorithm (BPA)—operate directly on time-domain or frequency-domain SAR measurements and serve as baseline reconstruction models for evaluating DIA performance.

For advanced reconstruction, we include the [Lightweight Imaging Algorithm (LIA)](https://www.mdpi.com/1424-8220/22/12/4509), an iterative, matrix-based method designed for efficient, high-quality imaging under irregular or non-uniform aperture trajectories.
***

### Files Required for Attack Implementation
The following dataset files must be placed in the same directory as the MATLAB attack script.
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


***

### Clean Results (with DIA)
The figure below shows the baseline reconstructions produced by the traditional SAR algorithms (MFA, RMA, BPA) and the advanced LIA method before applying any DIA. These clean outputs serve as ground-truth references for evaluating how the attack alters different reconstruction pipelines.

<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Traditional%20SAR%20Imaging%20Algos/files/MFA_RMA_BPA_LIA_clean.png"
     width="800" height="800">


***
### Atttack Results 
