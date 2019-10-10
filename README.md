# pytorch-timeseries

PyTorch implementations of deep neural neural nets for time series classification.

Currently, the following papers are implemented:
* [InceptionTime: Finding AlexNet for Time Series Classification](https://arxiv.org/abs/1909.04939)
* [Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline](https://arxiv.org/abs/1611.06455)

### Beyond the UCR/UEA archive
There are two ways use the Inception Time model on your own data:

1. Copy the [models](src/models), and write new training loops
2. Extend the [base trainer](src/trainer.py) by implementing an initializer, `get_loaders` and `save`. 
This allows the training code (which handles both single and multi-class outputs) to be used - an example of this is
the [`UCRTrainer`](src/ucr.py).

### Setup

[Anaconda](https://www.anaconda.com/download/#macos) running python 3.7 is used as the package manager. To get set up
with an environment, install Anaconda from the link above, and (from this directory) run

```bash
conda env create -f environment.yml
```
This will create an environment named `inception` with all the necessary packages to run the code. To 
activate this environment, run

```bash
conda activate inception
```

In addition, [UCR/UEA archive](https://www.cs.ucr.edu/~eamonn/time_series_data/) must be downloaded and stored in the 
[data folder](data).

### Scripts

Example scripts showing how to train and evaluate the model can be found in the [scripts folder](scripts).
