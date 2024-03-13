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


def preprocess_segmentation(data):
    # move to GPU and change data types
    # data = data.long()

    # create one-hot label map
    label_map = torch.LongTensor(data).unsqueeze(0)
    # *********************************************** important
    _, h, w = label_map.shape
    # h, w = label_map.shape
    nc = 8
    input_label = torch.FloatTensor(nc, h, w).zero_()
    input_semantics = input_label.scatter_(0, label_map, 1.0)

    return input_semantics


class LIDCDatasetTrain(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.data_dir = os.path.join("/home/zuxinrui/Desktop/TUDelft/Datasets/LIDC/", "image")
        self.data_list = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        item = dict()
        item["image"] = os.path.join("/home/zuxinrui/Desktop/TUDelft/Datasets/LIDC/image",
                                     self.data_list[i])
        item["label"] = os.path.join("/home/zuxinrui/Desktop/TUDelft/Datasets/LIDC/label",
                                     self.data_list[i][:-3] + "png")
        return item


class LIDCDatasetVal(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.data_dir = os.path.join("/home/zuxinrui/Desktop/TUDelft/Datasets/LIDC/test/", "image")
        self.data_list = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        item = dict()
        item["image"] = os.path.join("/home/zuxinrui/Desktop/TUDelft/Datasets/LIDC/test/image",
                                     self.data_list[i])
        item["label"] = os.path.join("/home/zuxinrui/Desktop/TUDelft/Datasets/LIDC/test/label",
                                     self.data_list[i][:-3] + "png")
        return item


class LIDCNoduleTrain(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.data_csv = pd.read_csv("data/LIDC-IDRI_Nodule_Properties.csv")
        self.data_list = self.data_csv["Slice"]  # attention: s should be capital...
        self.nodule_x_location = self.data_csv["x-position"]
        self.nodule_y_location = self.data_csv["y-position"]
        self.nodule_malignancy = self.data_csv["malignancy"]
        print("forming training dataset label list...")
        self.label_list = []
        for i in range(len(self.nodule_malignancy)):
            if self.nodule_malignancy[i] >= 3:
                self.label_list.append(1)
            else:
                self.label_list.append(0)
        print("training dataset label list formed.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        item = dict()
        item["image"] = os.path.join("/home/zuxinrui/Desktop/TUDelft/Datasets/LIDC/image", self.data_list[i] + ".jpg")
        item["label"] = os.path.join("/home/zuxinrui/Desktop/TUDelft/Datasets/LIDC/label", self.data_list[i] + ".png")
        item["x"] = self.nodule_x_location[i].astype("int32")
        item["y"] = self.nodule_y_location[i].astype("int32")
        item["malignancy"] = self.nodule_malignancy[i]
        item["features"] = self.data_csv.iloc[i]
        return item


class LIDCNoduleVal(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.data_csv = pd.read_csv("data/Nodule_Properties_Testset.csv")
        self.data_list = self.data_csv["slice"]
        self.nodule_x_location = self.data_csv["x_position_cs"]
        self.nodule_y_location = self.data_csv["y_position_cs"]
        self.nodule_malignancy = self.data_csv["malignancy"]
        print("forming training dataset label list...")
        self.label_list = []
        for i in range(len(self.nodule_malignancy)):
            if self.nodule_malignancy[i] >= 3:
                self.label_list.append(1)
            else:
                self.label_list.append(0)
        print("training dataset label list formed.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        item = dict()
        item["image"] = os.path.join("/home/zuxinrui/Desktop/TUDelft/Datasets/LIDC/test/image", self.data_list[i] + ".jpg")
        item["label"] = os.path.join("/home/zuxinrui/Desktop/TUDelft/Datasets/LIDC/test/label", self.data_list[i] + ".png")
        item["x"] = self.nodule_x_location[i].astype("int32")
        item["y"] = self.nodule_y_location[i].astype("int32")
        item["malignancy"] = self.nodule_malignancy[i]
        item["features"] = self.data_csv.iloc[i]
        return item
    

class LIDCNoduleTrainUI(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.data_csv = pd.read_csv("../data/LIDC-IDRI_Nodule_Properties.csv")
        self.data_list = self.data_csv["Slice"]  # attention: s should be capital...
        self.nodule_x_location = self.data_csv["x-position"]
        self.nodule_y_location = self.data_csv["y-position"]
        self.nodule_malignancy = self.data_csv["malignancy"]
        print("forming training dataset label list...")
        self.label_list = []
        for i in range(len(self.nodule_malignancy)):
            if self.nodule_malignancy[i] >= 3:
                self.label_list.append(1)
            else:
                self.label_list.append(0)
        print("training dataset label list formed.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        item = dict()
        item["image"] = os.path.join("/home/zuxinrui/Desktop/TUDelft/Datasets/LIDC/image", self.data_list[i] + ".jpg")
        item["label"] = os.path.join("/home/zuxinrui/Desktop/TUDelft/Datasets/LIDC/label", self.data_list[i] + ".png")
        item["x"] = self.nodule_x_location[i].astype("int32")
        item["y"] = self.nodule_y_location[i].astype("int32")
        item["malignancy"] = self.nodule_malignancy[i]
        item["features"] = self.data_csv.iloc[i]
        return item


class LIDCNoduleValUI(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.data_csv = pd.read_csv("../data/Nodule_Properties_Testset.csv")
        self.data_list = self.data_csv["slice"]
        self.nodule_x_location = self.data_csv["x_position_cs"]
        self.nodule_y_location = self.data_csv["y_position_cs"]
        self.nodule_malignancy = self.data_csv["malignancy"]
        print("forming training dataset label list...")
        self.label_list = []
        for i in range(len(self.nodule_malignancy)):
            if self.nodule_malignancy[i] >= 3:
                self.label_list.append(1)
            else:
                self.label_list.append(0)
        print("training dataset label list formed.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        item = dict()
        item["image"] = os.path.join("/home/zuxinrui/Desktop/TUDelft/Datasets/LIDC/test/image", self.data_list[i] + ".jpg")
        item["label"] = os.path.join("/home/zuxinrui/Desktop/TUDelft/Datasets/LIDC/test/label", self.data_list[i] + ".png")
        item["x"] = self.nodule_x_location[i].astype("int32")
        item["y"] = self.nodule_y_location[i].astype("int32")
        item["malignancy"] = self.nodule_malignancy[i]
        item["features"] = self.data_csv.iloc[i]
        return item


class LIDCDatasetCrop(Dataset):
    def __init__(self, size=None,
                 degradation=None, downscale_f=4):
        self.base = self.get_base()
        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.LR_size = int(size / downscale_f)

        # select cropper:
        # self.cropper = albumentations.RandomCrop(height=256, width=256)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        file = self.base[i]

        # image construction
        image = Image.open(file["image"])
        image = np.array(image).astype(np.uint8)
        image = self.cropper(image=image)["image"]

        # segmentation label construction
        label = Image.open(file["label"])
        label = np.array(label).astype(np.uint8)
        label_3c = np.zeros((label.shape[0], label.shape[1], 3))
        label_3c[:, :, 0] = label
        label_3c[:, :, 1] = label
        label_3c[:, :, 2] = label
        label = self.cropper(image=label_3c)["image"]

        x = dict()
        x["image"] = (image / 127.5 - 1.0).astype(np.float32)
        x["label"] = (label / 2 - 1.0).astype(np.float32)
        return x


class LIDCDatasetCropOneHot(Dataset):
    def __init__(self, size=None,
                 degradation=None, downscale_f=8, num_classes=8, concat=True):
        self.base = self.get_base()
        assert size
        assert (size / downscale_f).is_integer()
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
        # *********************************************** important ***********************************************
        _, h, w = label_map.shape
        # h, w = label_map.shape
        nc = self.num_classes
        input_label = torch.FloatTensor(nc, h, w).zero_()
        input_semantics = input_label.scatter_(0, label_map, 1.0)

        return input_semantics

    def __getitem__(self, i):

        # get base dataset item:
        file = self.base[i]

        # image construction:
        image = Image.open(file["image"])
        image = np.array(image).astype(np.uint8)
        image = image[:, :, 0]

        # segmentation map construction:
        label = Image.open(file["label"])
        label = np.array(label).astype(np.uint8)
        label_binary = self.preprocess_segmentation(label)

        # set up the item dictionary:
        x = dict()
        x["image"] = (image / 127.5 - 1.0).astype(np.float32)
        x["label"] = label_binary
        if self.concat:
            x["concat"] = np.expand_dims(label, axis=0).astype(np.float32)
        return x



class LIDCDatasetSingleCh(Dataset):
    def __init__(self, size=None):
        self.base = self.get_base()
        assert size
        self.size = size

        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        file = self.base[i]

        # image construction
        image = Image.open(file["image"])
        image = np.array(image).astype(np.uint8)
        # image = self.cropper(image=image)["image"]
        image = image[:, :, 0]

        # segmentation label construction
        label = Image.open(file["label"])
        label = np.array(label).astype(np.uint8)

        x = dict()
        x["image"] = (image / 127.5 - 1.0).astype(np.float32)
        x["label"] = (label / 2 - 1.0).astype(np.float32)
        return x


class LIDCDatasetClassifier(Dataset):
    def __init__(self, nodule_crop_size=None):
        self.base = self.get_base()
        assert nodule_crop_size
        self.size = nodule_crop_size
        self.cropper = albumentations.CropNonEmptyMaskIfExists(height=self.size, width=self.size,
                                                               ignore_values=[0, 1, 2, 3, 4, 5, 6])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        file = self.base[i]

        # image construction
        image = Image.open(file["image"])
        image = np.array(image).astype(np.uint8)

        # segmentation label construction
        label = Image.open(file["label"])
        label = np.array(label).astype(np.uint8)
        if np.where(label == 7)[0].any() == False:
            return None
        nodule_img = self.cropper(image=image, mask=label)["image"]

        # crop_side_len = 512 * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        # crop_side_len = int(crop_side_len)
        # self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)
        x = dict()
        x["image"] = (nodule_img / 127.5 - 1.0).astype(np.float32)
        # x["label"] = (label/2 - 1.0).astype(np.float32)
        return x


class LIDCNoduleMalignancyClassifierBalanced(Dataset):
    def __init__(self, training_set=True, nodule_crop_size=None, maxpooling_pixels=4, random_rotation=False,
                 masked=False, masked_guidance=False, return_original_label=False):
        self.base = self.get_base()
        assert nodule_crop_size
        self.size = np.int32(nodule_crop_size / 2)
        pooling_size = np.int32(maxpooling_pixels*2 + 1)
        self.max_pool = torch.nn.MaxPool2d(pooling_size, stride=1, padding=maxpooling_pixels)  # pooling size: 9
        self.cropper = albumentations.CropNonEmptyMaskIfExists(height=self.size, width=self.size,
                                                               ignore_values=[0, 1, 2, 3, 4, 5, 6])
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

    def __getitem__(self, i):
        # file = self.base[i]
        # image construction
        dynamic_i = i
        while True:
            try:
                file = self.base[dynamic_i]
                axis1 = file["x"]
                axis2 = file["y"]
                assert axis1 >= self.size and axis2 >= self.size and axis1 <= 512-self.size and axis2 <= 512-self.size, "size not big enough."
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
                assert cropped_image.shape[-1] == 2*self.size and cropped_image.shape[-2] == 2*self.size, "size doesn't match."

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
        x["label"] = preprocess_segmentation(label)

        if self.masked_guidance:
            x["mask"] = mask.astype(np.int32)
            x['position'] = [axis1, axis2]
            x["crop_size_half"] = np.int32(self.size)

        if file["malignancy"] >= 3:
            x["class_label"] = 1
        else:
            x["class_label"] = 0
        # x["label"] = (label/2 - 1.0).astype(np.float32)
        # if self.include_key_features:
        #     x["lobulation"] = file["lobulation"]
        #     x["spiculation"] = file["spiculation"]
        #     x["texture"] = file["texture"]
        #     x["margin"] = file["margin"]
        #     x["subtlety"] = file["subtlety"]
        #     x["sphericity"] = file["sphericity"]
        return x


class LIDCNoduleKeyFeatureClassifierBalanced(Dataset):
    """
    training_set(default:True): augmentation
    nodule_crop_size(default:64)
    maxpooling_pixels(default:4): max-pool the nodule areas with certain pixels

    """
    def __init__(self, training_set=True, nodule_crop_size=None, maxpooling_pixels=4, random_rotation=False,
                 masked=False, masked_guidance=False, key_feature="lobulation", return_original_label=False):
        self.base = self.get_base()  # base dataset
        assert nodule_crop_size
        self.size = np.uint16(nodule_crop_size / 2)
        pooling_size = np.int32(maxpooling_pixels*2 + 1)
        self.max_pool = torch.nn.MaxPool2d(pooling_size, stride=1, padding=maxpooling_pixels)  # pooling size: 9
        self.masked = masked
        self.random_rotation = random_rotation
        self.rotate = transforms.RandomRotation((-180, 180))
        self.flip = transforms.Compose([transforms.RandomVerticalFlip(),
                                        transforms.RandomHorizontalFlip(),
                                        ])

        self.training_set = training_set
        self.key_feature = key_feature
        # the certain column values:
        self.value_list = np.array(self.base.data_csv[self.key_feature])
        self.label_list = np.int32(self.value_list * 2 - 2)
        # used for resample the data when the sampling procedure failed:
        self.feature_value_id = []
        for value in np.linspace(1, 5, 9):
            self.certain_value_id = np.where(self.value_list == value)[0]
            self.feature_value_id.append(self.certain_value_id)
        self.masked_guidance = masked_guidance
        self.return_original_label = return_original_label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        # file = self.base[i]
        # image construction
        dynamic_i = i
        while True:
            try:
                file = self.base[dynamic_i]
                image = Image.open(file["image"])
                axis1 = file["x"]
                axis2 = file["y"]
                assert axis1 >= self.size and axis2 >= self.size, "size not big enough."
                image = np.array(image).astype(np.uint8)
                image = image[:, :, 0]

                # segmentation label construction
                label = Image.open(file["label"])
                label = np.array(label).astype(np.uint8)

                if self.masked:
                    label_mask = np.array(self.max_pool(torch.Tensor(label).unsqueeze(0))).astype(np.uint8)
                    image[np.where(label_mask[0] < 5)] = 0  # mask the image with a boundary (value: 5,6,7)
                    assert label_mask.max() >= 5, "seg map has no nodule semantic label."
                # print("seg map max: ", label_mask.max())
                if self.masked_guidance:
                    label_diffuse = np.array(self.max_pool(torch.Tensor(label).unsqueeze(0))).astype(np.uint8)
                    mask = np.zeros(image.shape)
                    mask[np.where(label_diffuse[0] >= 5)] = 1

                # crop:
                image = image[axis1 - self.size:axis1 + self.size, axis2 - self.size:axis2 + self.size]
                if self.training_set:
                    if self.random_rotation:
                        image = self.rotate(torch.Tensor(image).unsqueeze(0)).squeeze(0)
                    else:
                        image = self.flip(torch.Tensor(image).unsqueeze(0)).squeeze(0)
                image = np.array(image)
                assert image.shape[-1] == 2*self.size and image.shape[-2] == 2*self.size, "size doesn't match."

            except FileNotFoundError as msg:
                print(msg)
                label = self.label_list[i]
                # select an index array with certain value:
                dynamic_i = np.random.choice(self.feature_value_id[label])
                continue
            except AssertionError as msg:
                print(msg)
                label = self.label_list[i]
                dynamic_i = np.random.choice(self.feature_value_id[label])
                continue
            else:
                if i != dynamic_i:
                    print('load another {} file with same label'.format(dynamic_i))
                break

        x = dict()
        x['filename'] = file['image']
        # x['position'] = [axis1, axis2]
        x["image"] = (image / 127.5 - 1.0).astype(np.float32)
        x["class_label"] = file["features"][self.key_feature].astype(np.float32)

        if self.return_original_label:
            x["original_label"] = label

        if self.masked_guidance:
            x["mask"] = mask.astype(np.int32)
            x['position'] = [axis1, axis2]
            x["crop_size_half"] = np.int32(self.size)
            x["label"] = preprocess_segmentation(label)

        return x


class LIDCDatasetCropTrain(LIDCDatasetCrop):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = LIDCDatasetTrain()
        return dset


class LIDCDatasetCropValidation(LIDCDatasetCrop):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = LIDCDatasetVal()
        return dset


# used in SPADEUNet LSDM:
class LIDCDatasetCropOneHotTrain(LIDCDatasetCropOneHot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = LIDCDatasetTrain()
        return dset


class LIDCDatasetCropOneHotValidation(LIDCDatasetCropOneHot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = LIDCDatasetVal()
        return dset


# used in autoencoder:
class LIDCDatasetSingleChTrain(LIDCDatasetSingleCh):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = LIDCDatasetTrain()
        return dset


class LIDCDatasetSingleChValidation(LIDCDatasetSingleCh):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = LIDCDatasetVal()
        return dset


# used in classifier:
class LIDCNoduleMalignancyClassifierBalancedTrain(LIDCNoduleMalignancyClassifierBalanced):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = LIDCNoduleTrain()
        return dset


class LIDCNoduleMalignancyClassifierBalancedValidation(LIDCNoduleMalignancyClassifierBalanced):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = LIDCNoduleVal()
        return dset


# used in key feature classifier:
class LIDCNoduleKeyFeatureClassifierBalancedTrain(LIDCNoduleKeyFeatureClassifierBalanced):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = LIDCNoduleTrain()
        return dset


class LIDCNoduleKeyFeatureClassifierBalancedValidation(LIDCNoduleKeyFeatureClassifierBalanced):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = LIDCNoduleVal()
        return dset
    

# used in ui:
# used in classifier:
class LIDCUITrain(LIDCNoduleMalignancyClassifierBalanced):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = LIDCNoduleTrainUI()
        return dset


class LIDCUIValidation(LIDCNoduleMalignancyClassifierBalanced):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = LIDCNoduleValUI()
        return dset

