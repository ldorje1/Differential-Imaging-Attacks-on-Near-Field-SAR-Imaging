Our proposed differential imaging attack (DIA) is applied to three models (CV-Deep2S, Deep2S, Deep2S+) introduced in the paper: "[Efficient Physics-Based Learned Reconstruction Methods for Real-Time 3D Near-Field MIMO Radar Imaging](https://www.sciencedirect.com/science/article/abs/pii/S105120042300369X)".



### Three 3D U-Net-based Models 
#### CV-Deep2S (Complex-Valued Deep2S)

#### Deep2S 
Deep2S is a two-stage, physics-guided learned reconstruction network for 3D near-field MIMO radar imaging.
Stage 1: Applies the adjoint operator A^H to map raw complex measurements into an intermediate 3D image volume.

Stage 2: A 3D U-Net refines this intermediate image to suppress artifacts from sparse frequency and antenna sampling.
Deep2S operates on magnitude-only intermediate reconstructions, which the paper shows gives high SSIM and strong artifact suppression.
#### Deep2S+


***


***

###
