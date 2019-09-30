# inception-time-pytorch

A PyTorch implementation of [InceptionTime](https://arxiv.org/pdf/1909.04939.pdf), a convolutional neural network for
time series classification.

For more information, see the original [tensorflow implementation](https://github.com/hfawaz/InceptionTime).

### Beyond the UCR/UEA archive
There are two ways use the Inception Time model on your own data:

1. Copy the [model](inception/inception.py), and write a new training loop for it
2. Extend the [base trainer](inception/trainer.py) by implementing an initializer, `get_loaders` and `save`. 
This allows the training code (which handles both single and multi-class outputs) to be used - an example of this is
the [`UCRTrainer`](inception/ucr.py).

### Setup

[Anaconda](https://www.anaconda.com/download/#macos) running python 3.7 is used as the package manager. To get set up
with an environment, install Anaconda from the link above, and (from this directory) run

```bash
conda env create -f environment.yml
```
This will create an environment named `crop_yield_prediction` with all the necessary packages to run the code. To 
activate this environment, run

```bash
conda activate inception
```

In addition, [UCR/UEA archive](https://www.cs.ucr.edu/~eamonn/time_series_data/) must be downloaded and stored in the 
[data folder](data).

### Scripts

Example scripts showing how to train and evaluate the model can be found in the [scripts folder](scripts).
