# Differential Imaging (DI) Attacks on Near-Field SAR Imaging
### (Applied to CSA (SBRIM))

Our proposed Differential Imaging Attack (DIA) is applied to the sparse imaging method: Sparsity Bayesian
Recovery via Iterative Minimum (SBRIM) introduced in the paper: [Sparse autofocus via Bayesian learning iterative maximum and applied for LASAR 3-D imaging](https://ieeexplore.ieee.org/document/6875674 ). 

CSA method introduce in the paper is a sparse SAR imaging method that uses an explicit physics-based forward model 𝐻 and an iterative optimization solver (SBRIM) to reconstruct the scene. Instead of simple matched filtering, it searches for a reflectivity image that both fits the measured data and is sparse, which typically gives sharper targets and lower clutter/sidelobes, especially under limited or noisy apertures.

***
## Files Required for the Attack Implementation

Please download all files from the following Google Drive folder: 👉 [Google Drive data folder ](https://drive.google.com/drive/folders/1gymInr98iKLn37k7IIvvssIoM6Zd3r5P?usp=drive_link) and then run the [DIA_CSA_main](https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Compressed%20Sensing%20Algo/DIA_CSA_main.m) file.

| File Name                                   | Description |
|---------------------------------------------|-------------|
| `rawSAR.mat`                                | Raw SAR ADC data cube (`adcDataCube`, size: Nsamp × X_axis × Y_axis) used as the input to CSA |
| `trueImage_complex_CSA.mat`                 | Clean CSA complex reconstruction image (`trueImage_complx_csa`, size: B_pixels × A_pixels) |
| `desired_attacked_complex_CSA.mat`          | Target camouflage image for the CSA spoofing attack (`sar_camouflaged`) |
| `iqData_noAtk.mat`                          | Clean IQ measurements used to extract clean attack signatures |
| `iqData_Atk.mat`                            | Attacked IQ measurements used to construct the CSA attack dictionary |



***
## Clean Results (without DI-Attack)

<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Compressed%20Sensing%20Algo/images/clean_csa.png"
     width="600" height="600">



***
## Atttack Results 

<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Compressed%20Sensing%20Algo/images/attacked_csa.png"
     width="400" height="400">

