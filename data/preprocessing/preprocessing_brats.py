import os
import numpy as np
import SimpleITK as sitk
import cv2
from matplotlib import pyplot as plt
import pandas as pd
from skimage.filters import threshold_multiotsu
import matplotlib
from PIL import Image as im

from MRI_preprocessing_helper import nyul_train_standard_scale, nyul_apply_standard_scale


def readMhd_nifti(filename):
    """
    :param filename:
    :param gaze_time: if 'gaze_time'==True, script will keep the original dimensions of the scans (not always 512x512),
    in order to calculate the gaze_time correctly. If False, it will crop all images to 512x512
    :return:
    """
    # read mhd/raw image
    itkimage = sitk.ReadImage(filename)
    scan = sitk.GetArrayFromImage(itkimage)  # 3D image
    spacing = itkimage.GetSpacing()  # voxel size

    origin = itkimage.GetOrigin()  # world coordinates of origin
    transfmat = itkimage.GetDirection()  # 3D rotation matrix

    return scan, spacing, origin, transfmat


def create_semantic_map(slice_, thresh, image=True):
    if image:
        map_ = np.zeros(np.shape(slice_))
        map_[slice_ != 0] = 0
        map_[slice_ > thresh[0]] = 1
        map_[slice_ > thresh[1]] = 2
    else:
        map_ = np.zeros(np.shape(slice_))
        map_[slice_ == 1] = 3       # label 1 or necrotic core
        map_[slice_ == 4] = 4       # label 4 for GD enhanced area

    return map_


def get_target_location(slice_mask):
    contours, hierarchy = cv2.findContours(slice_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    multi = False
    if len(contours) == 0 or len(contours) > 1:
        # Indicate that there is either no glioma found, or more than one
        multi = True
    x_cent, y_cent = [], []
    for c in range(len(contours)):  # for every contour/nodule found
        if contours[c].shape[0] < 10:
            multi = True    # same as no glioma
            continue
        else:
            x, y, w, h = cv2.boundingRect(contours[c])
            #  turn into yolo coordinates (normalized xywh format)
            x_cent.append(x + (w / 2))
            y_cent.append(y + (h / 2))

    return x_cent, y_cent, multi


def check_seg(label_map):
    skip = False
    if np.count_nonzero(label_map[label_map == 1]) > np.count_nonzero(label_map[label_map == 2]):
        # threshold not satisfactory
        skip = True

    return skip


def save_images(im_scan, im_map, im_name, output_dir, my_dpi=600):
    # check if three folders exist and save image, label, put to that folder
    # save image slice
    output_fold = output_dir + '/image/'
    if not os.path.exists(output_fold):
        os.mkdir(output_fold)
    file_name = im_name + '.png'
    # 665 or my_dpi might need to be adjusted depending on your screen resolution
    plt.figure(figsize=(312 / my_dpi, 312 / my_dpi), dpi=my_dpi)
    #plt.imshow(im_scan, cmap=plt.cm.gray, vmin=im_scan.min(), vmax=im_scan.max())
    plt.imshow(im_scan, cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig(os.path.join(output_fold, file_name), bbox_inches='tight', pad_inches=0, dpi=my_dpi)
    plt.close()
    # Save semantic label map
    output_fold2 = output_dir + '/label/'
    if not os.path.exists(output_fold2):
        os.mkdir(output_fold2)
    label = im.fromarray(im_map)
    label = label.convert("L")
    label.save(output_fold2 + file_name)
    # Save semantic label map in color
    # cmap in manuscript review:
    #cmap = matplotlib.colors.ListedColormap(['#08094B', '#52648e', '#7ba4e9', '#86ac41', '#fbb41a'])
    # cmap in manuscript ps:
    cmap = matplotlib.colors.ListedColormap(['darkblue', 'slateblue', 'darkcyan', 'mediumaquamarine', 'yellow'])
    output_fold3 = output_dir + '/put/'
    if not os.path.exists(output_fold3):
        os.mkdir(output_fold3)
    # 665 or my_dpi might need to be adjusted depending on your screen resolution
    plt.figure(figsize=(312 / my_dpi, 312 / my_dpi), dpi=my_dpi)
    plt.imshow(im_map, cmap=cmap)
    plt.axis('off')
    plt.savefig(os.path.join(output_fold3, file_name), bbox_inches='tight', pad_inches=0, dpi=my_dpi)
    plt.close()


def process_scans(dir_in, dir_out, type_, path2norm, save=True):
    # Collect scans in input directory
    scans = os.listdir(dir_in + type_ + '/')
    # define which weighted mri image is necessary
    weight = 't1ce'     # t1 weighted contrast enhanced
    # create array for slices/scans we might skip.
    skips = list(['skip_these_slices'])

    for brats_scan in scans:
        # Read scan
        [scan_nonnorm, spacing, origin, transfmat] = readMhd_nifti(dir_in + type_ + '/' + brats_scan + '/' +
                                                                   brats_scan + '_' + weight + '.nii')
        print(spacing, origin, transfmat)

        # Read segmentation mask
        [mask, _, _, _] = readMhd_nifti(dir_in + type_ + '/' + brats_scan + '/' + brats_scan + '_seg' + '.nii')

        # for every slice in scan, save image, sem_map (color and normal) and x/y pos abnormality
        df = pd.DataFrame({'patient_id': [], 'glioma_type': [], 'pathology': [], 'image_name': [],
                           'y_position': [], 'x_position': []})

        # normalize scan
        scan = nyul_apply_standard_scale(scan_nonnorm, path2norm)

        # Get treshold for creating the mask from somewhere in the middle of the scan
        middle = int(np.round(np.shape(scan)[0] / 2))
        three_fourth = int(middle + (middle/2))
        # mask out glioma if in the slice, so thresholding does not take it into account
        mask_tf = mask[three_fourth]
        scan_tf = scan[three_fourth].copy()
        scan_tf[mask_tf != 0] = 0
        threshhold = threshold_multiotsu(scan_tf)

        for slice_num in range(0, scan.shape[0]):
            name = brats_scan + '_' + str(slice_num) + '_' + type_
            # only want to include label 1 and 4 in the mask
            labels_in_mask = np.unique(mask[slice_num])
            if 1 in labels_in_mask or 4 in labels_in_mask:

                # create semantic label map from brain image; background, white matter, gray matter, background?
                sem_map = create_semantic_map(scan[slice_num], thresh=threshhold, image=True)
                # turn mask into mask with 2 labels: Glioma area and non-glioma area
                ab_map = create_semantic_map(mask[slice_num], thresh=threshhold, image=False)
                # add maps together
                sem_map[ab_map == 3] = 3    # necrotic core
                sem_map[ab_map == 4] = 4    # GD enhanced area

                # check if wm and gm differentation is okay enough. If more gm, than save name and continue
                cont = check_seg(sem_map)
                if cont:
                    skips.append(name)
                    continue

                # Obtain x, and y center position of target
                x_pos, y_pos, not_one = get_target_location(ab_map)
                # skip the slice if no glioma is found, or more than one
                if not_one:
                    continue

                # save images
                if save:
                    save_images(scan[slice_num], sem_map, name, dir_out)

                # save dataframe data
                if type_ == 'HGG':
                    pathology = 1
                else:
                    pathology = 0
                df2 = pd.DataFrame({'patient_id': [brats_scan], 'glioma_type': [type_], 'pathology': [pathology],
                                    'image_name': [name], 'y_position': [x_pos[0]], 'x_position': [y_pos[0]]})
                df = pd.concat([df, df2], ignore_index=True)

            print('Finished slice {:03} out of {:03} from scan {}'.format(slice_num, scan.shape[0], brats_scan))

        # append dataframe of scan to existing csv file of gaze times
        df.to_csv((dir_out + 'BraTS_Lesion_Properties.csv'), mode='a', index=False, header=False)

    # save all skipped files to a csv
    df_skipped = pd.DataFrame({'skipped': [skips]})
    df_skipped.to_csv(dir_out + 'skipped_slices.csv')


if __name__ == "__main__":
    # Extract and display cube for example nodule
    output_f = 'D:/Details/datasets/BRATS2019/MICCAI_BraTS_2019_Data_Training/improved_labels/'
    input_f = 'D:/Details/datasets/BRATS2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/'
    standard_path = os.path.join(input_f, 'histograms/standard_test.npy')
    # Indicate if either HGG or LGG
    t_type = 'LGG'
    # Indicate if normalization histogram still needs to be obtained
    obtain_hist = False

    # check if csv exists
    if not os.path.exists(output_f + 'BraTS_Lesion_Properties.csv'):
        df_ = pd.DataFrame({'patient_id': [], 'glioma_type': [], 'pathology': [], 'image_name': [],
                           'y_position': [], 'x_position': []})
        df_.to_csv((output_f + 'BraTS_Lesion_Properties.csv'), index=False)

    if obtain_hist:
        # get list of all tc1e names and get standardization histogram
        hgg_scans = os.listdir(input_f + 'HGG/')
        lgg_scans = os.listdir(input_f + 'LGG/')
        all_scans = []
        ## add all files + location that end with t1ce.nii
        for folder in hgg_scans:
            scans_ = os.listdir(os.path.join(input_f, 'HGG', folder))
            for scan_names in scans_:
                if scan_names.endswith('t1ce.nii'):
                    all_scans.append(os.path.join(input_f, 'HGG', folder, scan_names))
        for folder in lgg_scans:
            scans_ = os.listdir(os.path.join(input_f, 'LGG', folder))
            for scan_names in scans_:
                if scan_names.endswith('t1ce.nii'):
                    all_scans.append(os.path.join(input_f, 'LGG', folder, scan_names))

        # Get standardized histogram for normalization
        standard_scale, perc = nyul_train_standard_scale(all_scans)
        # save standardized histogram
        standard_path = os.path.join(input_f, 'histograms/standard_test.npy')
        np.save(standard_path, [standard_scale, perc])

    process_scans(input_f, output_f, t_type, standard_path, save=True)



