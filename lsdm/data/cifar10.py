import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
import torch
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
import torchvision
import torchvision.transforms as transforms


class CIFAR10_OT_Train(Dataset):
    def __init__(self, config=None, size=32):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.size = size
        transform_train = transforms.Compose([transforms.ToTensor(),])
        self.dataset = torchvision.datasets.CIFAR10(root='/home/s2362341/torch/datasets/', train=True, transform=transform_train, download=True)
        self.ot_index = np.load("/home/s2362341/latent-diffusion/ldm/data/cifar_ot_indices.npy")
        self.source_data = np.load("/home/s2362341/pyOMT/gaussian_noise_fl32.npy")

    def __len__(self):
        return len(self.ot_index)

    def __getitem__(self, i):
        item = dict()
        image_target = np.array(self.dataset.__getitem__(i)[0].permute(1,2,0)).astype("float32")
        image_source = self.source_data[self.ot_index[i]].reshape((3,32,32))
        image_source = np.transpose(image_source, axes=(1,2,0))
        item["image"] = (image_target * 2 - 1).astype("float32")
        item["source"] = (image_source * 2 - 1).astype("float32")
        return item


class CIFAR10_OT_Validation(Dataset):
    def __init__(self, config=None, size=32):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.size = size
        transform_train = transforms.Compose([transforms.ToTensor(),])
        self.dataset = torchvision.datasets.CIFAR10(root='/home/s2362341/torch/datasets/', train=False, transform=transform_train, download=True)
        self.ot_index = np.load("/home/s2362341/latent-diffusion/ldm/data/cifar_ot_indices_val.npy")
        self.source_data = np.load("/home/s2362341/pyOMT/gaussian_noise_val.npy")

    def __len__(self):
        return len(self.ot_index)

    def __getitem__(self, i):
        item = dict()
        image_target = np.array(self.dataset.__getitem__(i)[0].permute(1,2,0)).astype("float32")
        image_source = self.source_data[self.ot_index[i]].reshape((3,32,32))
        image_source = np.transpose(image_source, axes=(1,2,0))
        item["image"] = (image_target * 2 - 1).astype("float32")
        item["source"] = (image_source * 2 - 1).astype("float32")
        return item

