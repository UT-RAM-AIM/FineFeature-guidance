# FineFeature-guidance
Code to our manuscript about the DiFine model for Fine Feature guidance in artificial medical images.

## Introduction

This repository contains the code to reproduce the results of our manuscript "Fine Feature guidance in artificial medical images using the DiFine model". The code is written in Python and uses the PyTorch library.

The DiFine model contains an autoencoder, a diffusion model with coarse feature guidance, and a classifier for fine feature guidance.

In the following, we will introduce the pre-request packages and the usage of the code.

## Pre-request

The code is written in Python 3.7 and uses the following packages:

- PyTorch 1.7.1
- torchvision 0.8.2
- numpy 1.19.2
- pytorch-lightning 1.1.4

## Usage

The code is organized in the following way:

- `configs/`: contains the configuration file
  - `autoencoder/`: contains the configuration file for the autoencoder
  - `latent-diffusion/`: contains the configuration file for the diffusion model
  - `classifier/`: contains the configuration file for the classifier
- `data/`: contains the data used in the experiments
- `models/`: contains the DiFine model
- `utils/`: contains the code to train and test the DiFine model
- `main.py`: contains the code to train and test the DiFine model

To train the DiFine model, you can run the following command:

```bash
python main.py --train
```

To test the DiFine model, you can run the following command:

```bash
python main.py --test
```

## Data

The data used in the experiments is stored in the `data/` folder. The data is organized in the following way:

- `data/`: contains the data used in the experiments
  - `train/`: contains the training data
  - `test/`: contains the test data
  - `val/`: contains the validation data
  - `train.csv`: contains the file names and labels of the training data
  - `test.csv`: contains the file names and labels of the test data


