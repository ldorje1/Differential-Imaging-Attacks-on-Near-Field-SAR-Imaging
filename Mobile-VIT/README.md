Our proposed differential imaging attack is applied to the Mobile-VIT model introduced in the paper:
[“A Vision Transformer Approach for Efficient Near-Field Irregular SAR Super-Resolution.”](https://arxiv.org/pdf/2305.02074)




- data/rawSAR.mat
    variable: adcDataCube   (Nsamp x M x N), complex
- data/D.mat
    variable: D             (Nsamp x (M*N)), complex
- data/desired_attacked_complex_MFA_RMA.mat
    variable: sar_camouflaged  (H x W), complex
- models/hffh_vit_best_epoch_050.pth
    trained Mobile-ViT checkpoint
