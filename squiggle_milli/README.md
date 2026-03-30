<!--
# Differential Imaging (DI) Attacks on Near-Field SAR Imaging
(Applied to SquiggleMilli) 

Our proposed differential imaging attack (DIA) is applied to the SquiggleMilli model, which uses cGAN achitecture, introduced in the paper: ["SquiggleMilli: Approximating SAR Imaging on Mobile Millimeter-Wave Devices"](https://dl.acm.org/doi/10.1145/3478113).

***
## 📁 Files Required for the Attack Implementation  



**📌 Note:**

(1) The pretrained SquiggleMilli weights provided in the repository appear to be corrupted. To ensure consistency, we re-trained the model from scratch using the original [dataset](https://www.dropbox.com/scl/fo/ibj6m0qcy0pmaqwltkncs/AORJc1tvXvlU1mHCwdCz2QE?rlkey=exu9kisonu5l2wozcpdh0d3cr&e=1&dl=0).

***
## Re-trained SquiggleMilli Clean Results
Below is the final part of the training log from our SquiggleMilli model retraining.  This confirms successful convergence and checkpoint saving during the last epochs:

```text
Epoch 293/300 | D loss: 0.5008 | G loss: 21.2852
Saved checkpoints for epoch 293 to models/
Epoch 294/300 | D loss: 0.5008 | G loss: 21.2618
Saved checkpoints for epoch 294 to models/
Epoch 295/300 | D loss: 0.5008 | G loss: 21.2437
Saved checkpoints for epoch 295 to models/
Epoch 296/300 | D loss: 0.5008 | G loss: 21.2278
Saved checkpoints for epoch 296 to models/
Epoch 297/300 | D loss: 0.5008 | G loss: 21.2104
Saved checkpoints for epoch 297 to models/
Epoch 298/300 | D loss: 0.5008 | G loss: 21.1962
Saved checkpoints for epoch 298 to models/
Epoch 299/300 | D loss: 0.5008 | G loss: 21.1856
Saved checkpoints for epoch 299 to models/
Epoch 300/300 | D loss: 0.5008 | G loss: 21.1712
Saved checkpoints for epoch 300 to models/
Training complete.
```





Our re-trained SquiggleMilli genetor results @
epoch 40, epoch 100, and epoch 300 

<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/SquiggleMilli/files/epoch_40_output.png"
     width="350" height="350">
 
<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/SquiggleMilli/files/epoch_100_output.png"
     width="350" height="350">
 
<img src="https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/SquiggleMilli/files/epoch_300_output.png"
     width="350" height="350">



***

### Attack Results 
-->
