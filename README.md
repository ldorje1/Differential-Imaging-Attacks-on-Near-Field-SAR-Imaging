# Adversarial Robustness of Millimeter-wave Imaging Algorithms

---
<!--
## Update on v2 (2/14/2025)
- Unified Platform: All models are now integrated into the VS Code environment, replacing the need for Google Colab or Anaconda/Jupyter Notebooks.
- Consistent Normalization: Implemented same normalization protocols across all MATLAB and Python models.
- Target Image Generation: Added functionality to generate target images using the model itself by processing randomly shuffled raw SAR data.
- Combined six variants of physics-based DNN models into a single main file to streamline adversarial attack implementation.
- Results: All preliminary attack results are now stored in the targeted_dia_results_1 directory.
---
-->
Work in Progress... apologies!

This repository contains the official implementation of  
**“Adversarial Robustness of Millimeter-wave Imaging Algorithms”**

#### 📁 All datasets and trained models (including those too large to host on GitHub) required for the attack implementation are available here: [data/files/trained-models](https://drive.google.com/drive/u/1/folders/1gymInr98iKLn37k7IIvvssIoM6Zd3r5P).

---

### Adversarial Millimeter-Wave testbed

<p align="center">
  <img src="figures/adv_test_bed.svg" alt="Figure 1" width="500"/>
</p>

---

### Visual comparison of imaging results under the proposed (DIA) target-conceal adversarial attack
<p align="center">
  <img src="figures/target_conceal_visual.png" alt="Figure 2" width="500"/>
</p>

---

### Visual comparison of imaging results under the proposed (DIA) target-swap adversarial attack
<p align="center">
  <img src="figures/target_swap_visual.png" alt="Figure 3" width="500"/>
</p>

---

### Visual comparison of imaging results under (DIA) the random weights adversarial attack
<p align="center">
  <img src="figures/random_w_visual.png" alt="Figure 4" width="500"/>
</p>
