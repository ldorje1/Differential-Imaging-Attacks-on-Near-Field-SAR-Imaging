Our proposed differential imaging attack is applied to the Mobile-VIT model introduced in the paper:
[“A Vision Transformer Approach for Efficient Near-Field Irregular SAR Super-Resolution.”](https://arxiv.org/pdf/2305.02074)

Since the original Mobile-ViT SAR model does not publicly provide trained weights,  
we re-trained the network following the methodology described in the paper.

Please use our provided checkpoint:

**`hffh_vit_best_epoch_050.pth`**

when running the differential imaging attack on Mobile-ViT.

https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/Mobile-VIT/files/mobile_vit_training_log.png

- data/rawSAR.mat
    variable: adcDataCube   (Nsamp x M x N), complex
- data/D.mat
    variable: D             (Nsamp x (M*N)), complex
- data/desired_attacked_complex_MFA_RMA.mat
    variable: sar_camouflaged  (H x W), complex
- models/hffh_vit_best_epoch_050.pth
    trained Mobile-ViT checkpoint
