import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
import pandas as pd


class BreastMamDatasetTrain(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.data_dir = os.path.join("/deepstore/datasets/ram/nodule-ct-gen/LSDM/BreastMam/train", "image")
        self.data_list = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        item = dict()
        item["image"] = os.path.join("/deepstore/datasets/ram/nodule-ct-gen/LSDM/BreastMam/train/image",
                                     self.data_list[i])
        item["label"] = os.path.join("/deepstore/datasets/ram/nodule-ct-gen/LSDM/BreastMam/train/label",
                                     self.data_list[i])
        return item


class BreastMamDatasetVal(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.data_dir = os.path.join("/deepstore/datasets/ram/nodule-ct-gen/LSDM/BreastMam/test", "image")
        self.data_list = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        item = dict()
        item["image"] = os.path.join("/deepstore/datasets/ram/nodule-ct-gen/LSDM/BreastMam/test/image",
                                     self.data_list[i])
        item["label"] = os.path.join("/deepstore/datasets/ram/nodule-ct-gen/LSDM/BreastMam/test/label",
                                     self.data_list[i])
        return item


# train LSDM diffusion:
class BreastMam(Dataset):
    def __init__(self, size=None, num_classes=3, concat=True):
        self.base = self.get_base()
        assert size
        self.size = size

        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.num_classes = num_classes
        self.concat = concat

    def __len__(self):
        return len(self.base)

    def preprocess_segmentation(self, data):
        # move to GPU and change data types
        # data = data.long()

        # create one-hot label map
        label_map = torch.LongTensor(data).unsqueeze(0)
        # *********************************************** important
        _, h, w = label_map.shape
        # h, w = label_map.shape
        nc = self.num_classes
        input_label = torch.FloatTensor(nc, h, w).zero_()
        input_semantics = input_label.scatter_(0, label_map, 1.0)

        return input_semantics

    def __getitem__(self, i):
        file = self.base[i]
        # image construction
        image = Image.open(file["image"])
        image = np.array(image).astype(np.uint8)
        image = image[:, :, 0]
        # image = self.cropper(image=image)["image"]

        # segmentation label construction
        label = Image.open(file["label"])
        label = np.array(label)

        label_one_hot = self.preprocess_segmentation(label)

        x = dict()
        x["image"] = (image / 127.5 - 1.0).astype(np.float32)
        x["label"] = label_one_hot
        if self.concat:
            # important!!!!
            label = (label - 1.0).astype(np.float32)
            x["concat"] = np.expand_dims(label, axis=0).astype(np.float32)
        return x


# inference / train classifier:
class BreastMamInference(Dataset):
    def __init__(self, training_set=True, nodule_crop_size=None, maxpooling_pixels=4, random_rotation=False,
                 masked=False, masked_guidance=False, return_original_label=False):
        self.base = self.get_base()
        assert nodule_crop_size
        self.size = np.int32(nodule_crop_size / 2)
        pooling_size = np.int32(maxpooling_pixels * 2 + 1)
        self.max_pool = torch.nn.MaxPool2d(pooling_size, stride=1, padding=maxpooling_pixels)  # pooling size: 9
        self.masked = masked
        self.random_rotation = random_rotation
        self.rotate = transforms.RandomRotation((-180, 180))
        self.flip = transforms.Compose([transforms.RandomVerticalFlip(),
                                        transforms.RandomHorizontalFlip(),
                                        ])

        self.label_list = np.array(self.base.label_list)
        self.training_set = training_set
        self.benign_id = np.where(self.label_list == 0)[0]
        self.malignant_id = np.where(self.label_list == 1)[0]
        self.masked_guidance = masked_guidance
        self.return_original_label = return_original_label

    def __len__(self):
        return len(self.base)

    def preprocess_segmentation(self, data):
        # move to GPU and change data types
        # data = data.long()

        # create one-hot label map
        label_map = torch.LongTensor(data).unsqueeze(0)
        # *********************************************** important
        _, h, w = label_map.shape
        # h, w = label_map.shape
        nc = self.num_classes
        input_label = torch.FloatTensor(nc, h, w).zero_()
        input_semantics = input_label.scatter_(0, label_map, 1.0)

        return input_semantics

    def __getitem__(self, i):
        # file = self.base[i]
        # image construction
        dynamic_i = i
        while True:
            try:
                file = self.base[dynamic_i]
                axis1 = file["x"]
                axis2 = file["y"]
                assert axis1 >= self.size and axis2 >= self.size and axis1 <= 512 - self.size and axis2 <= 512 - self.size, "size not big enough."
                image = Image.open(file["image"])
                image = np.array(image).astype(np.uint8)
                image = image[:, :, 0]

                # segmentation label construction
                label = Image.open(file["label"])
                label = np.array(label).astype(np.uint8)

                if self.masked:
                    label_diffuse = np.array(self.max_pool(torch.Tensor(label).unsqueeze(0))).astype(np.uint8)
                    image[np.where(label_diffuse[0] < 5)] = 0  # mask the image with a boundary (value: 5,6,7)
                    assert label_diffuse.max() >= 5, "seg map has no nodule semantic label."
                # print("seg map max: ", label_diffuse.max())

                if self.masked_guidance:
                    label_diffuse = np.array(self.max_pool(torch.Tensor(label).unsqueeze(0))).astype(np.uint8)
                    mask = np.zeros(image.shape)
                    mask[np.where(label_diffuse[0] >= 5)] = 1

                # crop:
                cropped_image = image[axis1 - self.size:axis1 + self.size, axis2 - self.size:axis2 + self.size]
                if self.training_set:
                    if self.random_rotation:
                        cropped_image = self.rotate(torch.Tensor(cropped_image).unsqueeze(0)).squeeze(0)
                    else:
                        cropped_image = self.flip(torch.Tensor(cropped_image).unsqueeze(0)).squeeze(0)
                cropped_image = np.array(cropped_image)
                assert cropped_image.shape[-1] == 2 * self.size and cropped_image.shape[
                    -2] == 2 * self.size, "size doesn't match."

            except FileNotFoundError as msg:
                print(msg)
                label = self.label_list[i]
                if label == 0:
                    dynamic_i = np.random.choice(self.benign_id)
                else:
                    dynamic_i = np.random.choice(self.malignant_id)
                continue
            except AssertionError as msg:
                print(msg)
                label = self.label_list[i]
                if label == 0:
                    dynamic_i = np.random.choice(self.benign_id)
                else:
                    dynamic_i = np.random.choice(self.malignant_id)
                continue
            else:
                if i != dynamic_i:
                    print('load another {} file with same label'.format(dynamic_i))
                break

        x = dict()
        x['filename'] = file['image']
        x["image"] = (cropped_image / 127.5 - 1.0).astype(np.float32)
        x["original_image"] = (image / 127.5 - 1.0).astype(np.float32)
        if self.return_original_label:
            x["original_label"] = label
        x["label"] = self.preprocess_segmentation(label)

        if self.masked_guidance:
            x["mask"] = mask.astype(np.int32)
            x['position'] = [axis1, axis2]
            x["crop_size_half"] = np.int32(self.size)

        if file["malignancy"] >= 3:
            x["class_label"] = 1
        else:
            x["class_label"] = 0

        return x


# used in SPADEUNet LSDM:
class BreastMamTrain(BreastMam):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = BreastMamDatasetTrain()
        return dset


class BreastMamValidation(BreastMam):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = BreastMamDatasetVal()
        return dset