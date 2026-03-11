This study proposes a data-driven surrogate modeling framework for fast pointwise 
prediction of subsonic airfoil flow fields at high Reynolds numbers. The framework 
combines a residual autoencoder for compact airfoil geometry representation with a 
feature-enhanced residual-attention multilayer perceptron (FE-RAMLP) for pointwise 
flow field reconstruction. The residual autoencoder compresses two-dimensional airfoil 
geometries into a four-dimensional latent space, which is shown to preserve geometric 
similarity across a large airfoil dataset. To address the degraded performance of 
conventional pointwise surrogate models at high Reynolds numbers, particularly in 
near-wall regions with strong gradients and under sparsely sampled training conditions, 
FE-RAMLP incorporates unsigned distance information and multi-scale Fourier feature 
encoding as feature enhancement mechanisms. Quantitative evaluations based on mean 
absolute error demonstrate that FE-RAMLP consistently achieves low prediction errors 
across all evaluated cases. For representative test cases, compared with a baseline 
residual-attention multilayer perceptron, the proposed model reduces the near-wall 
prediction error by more than 50%. In terms of computational efficiency, FE-RAMLP 
reconstructs full flow fields within approximately one second per case, achieving nearly 
three orders of magnitude speedup compared with CFD simulations of comparable 
resolution. Additional analyses based on flow field visualizations and near-wall 
distributions indicate that FE-RAMLP accurately captures complex flow structures 
with good physical consistency. Overall, the proposed framework provides an efficient 
and robust pointwise surrogate modeling approach for high-Reynolds-number subsonic 
airfoil flows, and is well suited for rapid aerodynamic evaluation and parametric studies 
in aerospace engineering applications. 

1. Compress airfoil binary images using an autoencoder to extract 4D geometric features.
The architecture of the Autoencoder model is shown below:
![github_qe](https://github.com/user-attachments/assets/d79b6141-2e6c-44a1-a4c5-cd5800b0487a)

2. The 4D geometric features, together with spatial coordinates (x, y) and flow conditions (Ma, AOA), are fed into the FE-RAMLP surrogate model to predict the velocity components (U, V) and pressure coefficient (Cp).
The architecture of the FE-RAMLP surrogate is shown below:
![feramlp](https://github.com/user-attachments/assets/f2c85ccc-587e-4762-abdb-f609dfd84da6)

For the implementation and training of the AE model, please refer to the RAMLP section.

For the implementation and training of FERAMLP, please refer to Main_model_train.py.



