# Helper functions of the preprocessing_Mammo.py code
import numpy as np
from skimage import measure, morphology
import cv2


def remove_noise_artifacts(img_, threshold=0.1, disk_size=18):
    # create binary image
    bin_img_ = np.zeros(img_.shape, np.uint8)
    bin_img_[img_ >= threshold] = 1
    # make contours smoother and remove noise by erosion and dilation
    ftprint = morphology.disk(disk_size)
    eroded = morphology.erosion(bin_img_, ftprint)
    dilated = morphology.dilation(eroded, ftprint)
    # get groups left in image
    label_image = measure.label(dilated)
    # find unique groups and size of the groups (aka counts)
    unique, counts = np.unique(label_image, return_counts=True)
    # sometimes the background is largest count, so need to check average value per index
    values = []
    for i in unique:
        sub = np.copy(img_)
        values.append(np.mean(sub[label_image == i]))
    values = np.asarray(values)
    # only take into account counts with more than 1000000? Otherwise might take into account artefacts
    indices = counts > 1000000
    values[indices == False] = 0
    # get index of largest mean
    index = np.asarray(values).argmax()
    # create mask
    mask = np.zeros(bin_img_.shape, np.uint8)
    mask[label_image == index] = 1
    # apply mask to image
    new_img = img_.copy()
    new_img[mask == 0] = 0

    return new_img, mask


def flip_image(image, img_mask):
    rows, cols = img_mask.shape
    xc = cols // 2
    # sum columns
    sum_cols = img_mask.sum(axis=0)
    sum_left = sum(sum_cols[0:xc])
    sum_right = sum(sum_cols[xc:-1])
    # flip if necessary
    if sum_left < sum_right:
        new_im = np.fliplr(image)  # flip image
        new_mask = np.fliplr(img_mask)
        flip = True
    else:
        new_im = image
        new_mask = img_mask
        flip = False

    return new_im, new_mask, flip


def pad2square(im):
    rows, cols = im.shape
    if rows != cols:
        # get the longest side
        if cols < rows:
            new_shape = (rows, rows)
        elif rows < cols:
            new_shape = (cols, cols)

        # pad image
        img_pad_ = np.zeros(shape=new_shape)
        img_pad_[:rows, :cols] = im
    else:
        img_pad_ = im

    return img_pad_


def clahe(img, clip=2.0, tile=(8, 8)):
    # contrast limited adaptive histogram equalisation (CLAHE); to improve contrast
    # Convert to uint8.
    # img = skimage.img_as_ubyte(img)
    img = cv2.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    img_uint8 = img.astype("uint8")

    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    clahe_img = clahe_create.apply(img_uint8)

    return clahe_img
