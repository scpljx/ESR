# Improve Adversarial Robustness via $\alpha$-EntMax Separation and Reactivation
Pytorch implementation of eSR.

[Improve Adversarial Robustness via $\alpha$-EntMax Separation and Reactivation]

## Updates
[05/2023] Code is released.

## $\alpha$-EntMax Separation and Reactivation
> (a) This work introduces a significant improvement in the feature separation mechanism by incorporating α-EntMax instead of traditional binary masks. This enhancement not only increases the flexibility in managing feature sparsity but also bolsters the model’s ability to identify and utilize robust features    through adaptive sparsity
>(b) By employing the Binary Cross-Entropy (BCE with Logits) loss function, the training process of the model is finely tuned, specifically targeting the optimization of separated features and their reactivated relationships. This not only boosts the model’s adaptability in dealing with complex adversarial samples but also enhances prediction accuracy by optimizing the recovery of non-robust features.


## Setup
The packages necessary for running our code are provided in `environment.yml`. Create the conda environment `ESR` by running:
```
conda env create -f environment.yml
```
Note that if you're using more latest GPUs (e.g., RTX 3090), you may need to refer to [this pytorch link](https://pytorch.org/get-started/locally/) to install the PyTorch package that suits your cuda version.

### Training
The codes for training our FSR module can be found in `train.py`. 


### Testing
After training, the model weights will be saved in `weights/[dataset]/[model]/[load_name].pth`. 

