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


"""
part 1: base dataset classes for autoencoders and diffusion models (separated into train and val)
"""

# dataset_dir = "/deepstore/datasets/ram/nodule-ct-gen/LSDM/BreastMam"

# please make sure that the image and label are in the same order:
class DatasetTrain(Dataset):
    def __init__(self, config=None, dataset_dir=None):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.image_dir = os.path.join(dataset_dir, "train", "image")
        self.label_dir = os.path.join(dataset_dir, "train", "label")
        self.name_list = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, i):
        item = dict()
        item["image"] = os.path.join(self.image_dir, self.name_list[i])
        item["label"] = os.path.join(self.label_dir, self.name_list[i])
        return item


class DatasetVal(Dataset):
    def __init__(self, config=None, dataset_dir=None):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.image_dir = os.path.join(dataset_dir, "test", "image")
        self.label_dir = os.path.join(dataset_dir, "test", "label")
        self.name_list = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, i):
        item = dict()
        item["image"] = os.path.join(self.image_dir, self.name_list[i])
        item["label"] = os.path.join(self.label_dir, self.name_list[i])
        return item


"""
part 2: base dataset classes for classifiers (separated into train and val)
"""

class DatasetCropTrain(Dataset):
    def __init__(self,
                 dataset_dir=None,
                 config=None,
                 data_csv_name=None,
                 feature_name=None,
                 slice_name=None,
                 make_binary=True,
                 positive_theshold=1,
                 ):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.image_dir = os.path.join(dataset_dir, "train", "image")
        self.label_dir = os.path.join(dataset_dir, "train", "label")
        self.data_csv_dir = os.path.join(data_csv_name)
        self.data_csv = pd.read_csv(self.data_csv_dir)
        self.name_list = self.data_csv[slice_name]  # attention: s should be capital...

        self.x_position = self.data_csv["x_position"]
        self.y_position = self.data_csv["y_position"]
        self.feature_degree = self.data_csv[feature_name]

        if make_binary:
            # use binary classification:
            print("[LSDM] forming training dataset binary label list...")
            self.label_list = []
            for i in range(len(self.feature_degree)):
                if self.feature_degree[i] >= positive_theshold:
                    self.label_list.append(1)
                else:
                    self.label_list.append(0)
            print("training dataset binary label list formed.")
        else:
            # better use regression model:
            self.label_list = self.feature_degree

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, i):
        item = dict()
        item["image"] = os.path.join(self.image_dir, self.name_list[i] + ".png")
        item["label"] = os.path.join(self.label_dir, self.name_list[i] + ".png")
        item["x"] = self.x_position[i].astype("int32")
        item["y"] = self.y_position[i].astype("int32")
        item["degree"] = self.label_list[i]
        item["features"] = self.data_csv.iloc[i]  # whole row
        return item


class DatasetCropVal(Dataset):
    def __init__(self,
                 dataset_dir=None,
                 config=None,
                 data_csv_name=None,
                 feature_name=None,
                 slice_name=None,
                 make_binary=True,
                 positive_theshold=1,
                 ):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.image_dir = os.path.join(dataset_dir, "test", "image")
        self.label_dir = os.path.join(dataset_dir, "test", "label")
        self.data_csv_dir = os.path.join(data_csv_name)
        self.data_csv = pd.read_csv(self.data_csv_dir)
        self.name_list = self.data_csv[slice_name]  # attention: s should be capital...

        self.x_position = self.data_csv["x_position"]
        self.y_position = self.data_csv["y_position"]
        self.feature_degree = self.data_csv[feature_name]

        if make_binary:
            # use binary classification:
            print("[LSDM] forming validation dataset binary label list...")
            self.label_list = []
            for i in range(len(self.feature_degree)):
                if self.feature_degree[i] >= positive_theshold:
                    self.label_list.append(1)
                else:
                    self.label_list.append(0)
            print("validation dataset binary label list formed.")
        else:
            # better use regression model:
            self.label_list = self.feature_degree

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, i):
        item = dict()
        item["image"] = os.path.join(self.image_dir, self.name_list[i] + ".png")
        item["label"] = os.path.join(self.label_dir, self.name_list[i] + ".png")
        item["x"] = self.x_position[i].astype("int32")
        item["y"] = self.y_position[i].astype("int32")
        item["degree"] = self.label_list[i]
        item["features"] = self.data_csv.iloc[i]  # whole row
        return item


"""
part 3: core dataset class for autoencoder / diffusion models:
"""

class DatasetGeneral(Dataset):
    def __init__(self,
                 size=None,
                 num_semantic_labels=3,
                 *args, **kwargs,
                 ):
        torch.cuda.empty_cache()
        self.base = self.get_base()
        assert size
        self.size = size

        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.num_semantic_labels = num_semantic_labels

    def __len__(self):
        return len(self.base)

    def preprocess_segmentation(self, data):
        label_map = torch.LongTensor(data).unsqueeze(0)
        # ********************** important
        _, h, w = label_map.shape
        # h, w = label_map.shape
        nc = self.num_semantic_labels
        input_label = torch.FloatTensor(nc, h, w).zero_()
        input_semantics = input_label.scatter_(0, label_map, 1.0)

        return input_semantics

    def __getitem__(self, i):
        file = self.base[i]
        # image construction
        image = Image.open(file["image"])
        image = np.array(image).astype(np.uint8)
        image = self.cropper(image=image)["image"]
        # select one channel:
        image = image[:, :, 0]
        # print(image.shape)

        # segmentation label construction
        label = Image.open(file["label"])
        if label.size != image.shape:
            label = label.resize(image.shape, Image.NEAREST)
        label = np.array(label)
        # if the label is single channel (n*n):
        if len(label.shape) == 2:
            label = np.transpose(np.tile(label, (3, 1, 1)), (1, 2, 0))
        assert label.shape[-1] == 3, "label channel number not equal to 3."
        assert len(label.shape) == 3, "label shape not equal to 3."
        label = self.cropper(image=label)["image"][:,:,0]

        x = dict()
        # image: [-1,1]:
        x["image"] = (image / 127.5 - 1.0).astype(np.float32)
        # binary label: {0,1}(for segmentation in SPADE layers):
        x["label"] = self.preprocess_segmentation(label)
        # seg label: [-1,1]:
        label_norm = (label - label.min()) / (label.max() - label.min())
        label = np.array((label_norm * 2 - 1)).astype(np.float32)
        x["concat"] = np.expand_dims(label, axis=0).astype(np.float32)
        return x

"""
part 4: core dataset class for classifiers:
"""
class DatasetWithClassifierGeneral(Dataset):
    def __init__(self,
                 training_mode=True,
                 crop_size=None,
                 random_rotation=False,
                 masked_image=False,
                 masked_guidance=False,
                 mask_maxpooling_pixels=4,
                 return_original_label=False,
                 num_semantic_labels=3,
                 abnormal_area_threshold=2,
                 label_scale=1,
                 *args, **kwargs,
                 ):
        # basic parameters:
        self.base = self.get_base()  # virtual function
        assert crop_size, "crop size not specified."
        self.size = np.int32(crop_size / 2)
        self.num_semantic_labels = num_semantic_labels
        pooling_size = np.int32(mask_maxpooling_pixels*2 + 1)
        self.max_pool = torch.nn.MaxPool2d(pooling_size, stride=1, padding=mask_maxpooling_pixels)  # pooling size: 9
        self.rotate = transforms.RandomRotation((-180, 180))
        self.flip = transforms.Compose([transforms.RandomVerticalFlip(),
                                        transforms.RandomHorizontalFlip(),
                                        ])

        self.label_list = np.array(self.base.label_list)
        self.training_mode = training_mode
        self.feature_degree_0_id = np.where(self.label_list == 0)[0]
        self.feature_degree_1_id = np.where(self.label_list == 1)[0]
        # index for each degree label:
        self.feature_degree_ids = []
        for i in range(np.int16(self.label_list.max())+1):
            self.feature_degree_ids.append(np.int64(np.where(self.label_list == i)[0]))

        self.random_rotation = random_rotation
        self.masked_image = masked_image
        self.masked_guidance = masked_guidance
        self.return_original_label = return_original_label
        self.abnormal_area_threshold = abnormal_area_threshold
        self.label_scale = label_scale

    def __len__(self):
        return len(self.base)

    def preprocess_segmentation(self, data):
        label_map = torch.LongTensor(data).unsqueeze(0)
        # ********************** important
        _, h, w = label_map.shape
        # h, w = label_map.shape
        nc = self.num_semantic_labels
        input_label = torch.FloatTensor(nc, h, w).zero_()
        input_semantics = input_label.scatter_(0, label_map, 1.0)

        return input_semantics

    def __getitem__(self, i):

        # image construction
        # dynamic index:
        dynamic_i = i
        while True:
            try:

                #### image loading and preprocessing loop:

                file = self.base[dynamic_i]
                axis1 = file["x"] * self.label_scale
                axis2 = file["y"] * self.label_scale

                assert axis1 >= self.size and axis2 >= self.size and axis1 <= 512-self.size and axis2 <= 512-self.size, "abnormal area key point out of boundary."

                image = Image.open(file["image"])
                image = np.array(image).astype(np.uint8)
                image = image[:, :, 0]

                # segmentation label construction
                label = Image.open(file["label"])
                if label.size != image.shape:
                    assert self.label_scale != 1, "should set label scale not equal to 1 for uneven label size"
                    label = label.resize(image.shape, Image.NEAREST)
                label = np.array(label).astype(np.uint8)

                # mask the image with a boundary (value: 5,6,7):
                if self.masked_image:
                    label_diffuse = np.array(self.max_pool(torch.Tensor(label).unsqueeze(0))).astype(np.uint8)
                    image[np.where(label_diffuse[0] < self.abnormal_area_threshold)] = 0  # mask the image with a boundary (value: 5,6,7)

                    assert label_diffuse.max() >= self.abnormal_area_threshold, "seg map has no abnormal area label."

                # provide a 'mask' key in the dictionary (for masked guidance):
                if self.masked_guidance:
                    label_diffuse = np.array(self.max_pool(torch.Tensor(label).unsqueeze(0))).astype(np.uint8)
                    mask = np.zeros(image.shape)
                    mask[np.where(label_diffuse[0] >= self.abnormal_area_threshold)] = 1

                # crop:
                cropped_image = image[axis1 - self.size:axis1 + self.size, axis2 - self.size:axis2 + self.size]
                # random rotation or flip when training:
                if self.training_mode:
                    if self.random_rotation:
                        cropped_image = self.rotate(torch.Tensor(cropped_image).unsqueeze(0)).squeeze(0)
                    else:
                        cropped_image = self.flip(torch.Tensor(cropped_image).unsqueeze(0)).squeeze(0)
                cropped_image = np.array(cropped_image)

                assert cropped_image.shape[-1] == 2*self.size and cropped_image.shape[-2] == 2*self.size, " cropped size doesn't match."

            # several exceptions and corresponding solutions (back to the loop):
            ## exception 1: several images are missing:
            except FileNotFoundError as msg:
                print(msg)
                # anchor the normal index to balance the binary labels:
                label = np.int16(self.label_list[i])
                dynamic_i = np.random.choice(self.feature_degree_ids[label])
                continue
            ## exception 2: the cropping process is not successful:
            except AssertionError as msg:
                print(msg)
                label = np.int16(self.label_list[i])
                dynamic_i = np.random.choice(self.feature_degree_ids[label])
                continue
            else:
                # after successfully loading the image, check whether the index remains and break the loop:
                if i != dynamic_i:
                    print('load another {} file with same label'.format(dynamic_i))
                break

        x = dict()
        x['filename'] = file['image']

        x["image"] = (cropped_image / 127.5 - 1.0).astype(np.float32)
        x["original_image"] = (image / 127.5 - 1.0).astype(np.float32)

        if self.return_original_label:
            x["original_label"] = label

        # label = self.preprocess_segmentation(label)
        x["label"] = self.preprocess_segmentation(label)
        # label_norm = (label - label.min()) / (label.max() - label.min())  # TODO: use normalized label as c_spade??
        # label = np.array((label_norm * 2 - 1)).astype(np.float32)
        # x["label"] = label
        label_norm = (label - label.min()) / (label.max() - label.min())
        label = np.array((label_norm * 2 - 1)).astype(np.float32)
        x["concat"] = np.expand_dims(label, axis=0).astype(np.float32)

        if self.masked_guidance:
            x["mask"] = mask.astype(np.int32)
            x['position'] = [axis1, axis2]
            x["crop_size_half"] = np.int32(self.size)

        x["class_label"] = np.int64(file["degree"])  # Long for CrossEntropyLoss

        return x


"""
part 5: final class for autoencoders and diffusion models (separated into train and val):
"""


class DatasetAPITrain(DatasetGeneral):
    """
        :parameter:
        1. training_mode: bool, whether to use random rotation and flip.
        2. crop_size: int, the size of the cropped image.
        """
    def __init__(self, dataset_dir=None, *args, **kwargs):
        self.dataset_dir = dataset_dir
        super().__init__(*args, **kwargs)

    # This function details the template of the father class <get_base>, like in C++ (even if there is no virtual function in father class):
    def get_base(self):
        dset = DatasetTrain(dataset_dir=self.dataset_dir)
        return dset


class DatasetAPIValidation(DatasetGeneral):
    """
        :parameter:
        1. training_mode: bool, whether to use random rotation and flip.
        2. crop_size: int, the size of the cropped image.
        """
    def __init__(self, dataset_dir=None, *args, **kwargs):
        self.dataset_dir = dataset_dir
        super().__init__(*args, **kwargs)

    # This function details the template of the father class <get_base>, like in C++ (even if there is no virtual function in father class):
    def get_base(self):
        dset = DatasetVal(dataset_dir=self.dataset_dir)
        return dset


"""
part 6: final class for classifiers (separated into train and val):
"""


class DatasetWithClassifierAPITrain(DatasetWithClassifierGeneral):
    """
        :parameter:
        1. training_mode: bool, whether to use random rotation and flip.
        2. crop_size: int, the size of the cropped image.
    """
    def __init__(self, 
                 dataset_dir=None, 
                 data_csv_name=None, 
                 feature_name=None, 
                 slice_name=None, 
                 make_binary=False, 
                 positive_theshold=1, 
                 *args, **kwargs):
        # defined by the base dataset class (only load strings):
        self.dataset_dir = dataset_dir
        self.data_csv_name = data_csv_name
        self.feature_name = feature_name
        self.make_binary = make_binary
        self.slice_name = slice_name
        self.positive_theshold = positive_theshold
        super().__init__(*args, **kwargs)

    # This function details the template of the father class <get_base>, like in C++ (even if there is no virtual function in father class):
    def get_base(self):
        dset = DatasetCropTrain(dataset_dir=self.dataset_dir, 
                                data_csv_name=self.data_csv_name, 
                                feature_name=self.feature_name, 
                                slice_name=self.slice_name, 
                                make_binary=self.make_binary,
                                positive_theshold=self.positive_theshold,
                                )
        return dset


class DatasetWithClassifierAPIValidation(DatasetWithClassifierGeneral):
    """
        :parameter:
        1. training_mode: bool, whether to use random rotation and flip.
        2. crop_size: int, the size of the cropped image.
        """
    def __init__(self, 
                 dataset_dir=None, 
                 data_csv_name=None, 
                 feature_name=None, 
                 slice_name=None, 
                 make_binary=False, 
                 positive_theshold=1, 
                 *args, **kwargs):
        # defined by the base dataset class (only load strings):
        self.dataset_dir = dataset_dir
        self.data_csv_name = data_csv_name
        self.feature_name = feature_name
        self.make_binary = make_binary
        self.slice_name = slice_name
        self.positive_theshold = positive_theshold
        super().__init__(*args, **kwargs)

    # This function details the template of the father class <get_base>, like in C++ (even if there is no virtual function in father class):
    def get_base(self):
        dset = DatasetCropVal(dataset_dir=self.dataset_dir, 
                              data_csv_name=self.data_csv_name, 
                              feature_name=self.feature_name, 
                              slice_name=self.slice_name, 
                              make_binary=self.make_binary,
                              positive_theshold=self.positive_theshold,
                              )
        return dset
