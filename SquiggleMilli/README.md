### SquiggleMilli 

Our proposed differential imaging attack (DIA) is applied to the SquiggleMilli model, which uses GAN achitecture, introduced in the paper:  
["SquiggleMilli: Approximating SAR Imaging on Mobile Millimeter-Wave Devices"](https://dl.acm.org/doi/10.1145/3478113).

***
### Files required for attack implementation

***
### Training Proof
Below is the final part of the training log from our SquiggleMilli model retraining.  This confirms successful convergence and checkpoint saving during the last epochs:

```text
Saved checkpoints for epoch 97 to models/
Epoch 98/100 | D loss: 0.5103 | G loss: 36.8397
Saved checkpoints for epoch 98 to models/
Epoch 99/100 | D loss: 0.5102 | G loss: 36.6664
Saved checkpoints for epoch 99 to models/
Epoch 100/100 | D loss: 0.5101 | G loss: 36.5011
Saved checkpoints for epoch 100 to models/
Training complete.
```

***
### Clean Results (without DIA)
SquiggleMilli genetor results 

![squiggle_epoch_60](https://github.com/ldorje1/Differential-Imaging-Attacks-on-Near-Field-SAR-Imaging/blob/main/SquiggleMilli/files/epoch_60_output.png)



***

### Attack Results 
