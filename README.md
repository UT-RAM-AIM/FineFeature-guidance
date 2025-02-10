# FineFeature-guidance
Code used for our manuscript "It’s All in the Details: Guiding Fine-Feature Characteristics in Artificial Medical Images using Diffusion Models" (submitted for publication).

## Introduction

The code is written in Python and uses the PyTorch library.

The DiFine model contains an autoencoder, a diffusion model with coarse feature guidance, and a classifier for fine feature guidance.

In the following, we will introduce the prerequisites packages and the usage of the code.

## Prerequisites

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
- `data/`: contains the feature info files (.csv) for the datasets
  - `preprocessing/`: contains the files for preprocessing of the used datasets
- `logs/`: used to store the training logs, each of which is stored in a separate folder named by the current time
- `lsdm/`: includes all the code for the DiFine model
- `main.py`: contains the code to train and test the DiFine model

To simply test the training procedure of the DiFine model, you can run the following command (on BrainMRI dataset):

```bash
python main.py --base configs/latent-diffusion/lsdm_general-BrainMRI3.yaml -t --gpus 0,
```

For specific training settings, please follow the instructions in the jupyter notebook `05-training.ipynb`.

To test the DiFine model, use `01/02/03/04-*.ipynb` to load the trained model and test the performance:

- `01-generation-LIDC-IDRI.ipynb`: test the DiFine model on the LIDC-IDRI dataset and visualize the classifier guidance
- `02-generation-BreastMam.ipynb`: test the DiFine model on the BreastMam dataset
- `03-generation-BrainMRI.ipynb`: test the DiFine model on the BrainMRI dataset
- `04-manipulation-LIDC-IDRI.ipynb`: test the manipulation ability of the DiFine model on the LIDC-IDRI dataset

Remember to check and change the parameters high-lighted on the top of each cell.

## Citation

TBD (manuscript submitted for publication)

The code in this repository is based on the previous work, and therefore their code, of:
1. Rombach, R., Blattmann, A., Lorenz, D., Esser, P. & Ommer, B. High-Resolution Image Synthesis with Latent Diffusion Models. in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) vols 2022-June 10674–10685 (IEEE, 2022).  (their [code](https://github.com/CompVis/latent-diffusion) is the basis of the used model)
2. Dhariwal, P. & Nichol, A. Diffusion Models Beat GANs on Image Synthesis. in Advances in Neural Information Processing Systems (eds. Ranzato;, M., Beygelzimer;, A., Dauphin;, Y., Liang;, P. S. & Vaughan;, J. W.) vol. 34 8780–8794 (Curran Associates, Inc., 2021).
3. Wang, W. et al. Semantic Image Synthesis via Diffusion Models. (2022) doi:10.48550/arXiv.2207.00050.

