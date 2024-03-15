import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
from skimage import transform
from PIL import Image as im
import pandas as pd
import cv2

from helper_pp_mammo import remove_noise_artifacts, flip_image, pad2square, clahe


def init_(input_folder_, patient_id_):
    k_ = patient_id_
    patients_ = input_folder_ + k_
    date = os.listdir(patients_ + '/')
    date.sort()

    # To pick a specific scan (date)
    for k in range(0, len(date)):
        # Skip if encounter non-folder objects
        if date[k].startswith('.'):
            continue
        else:
            date_ = patients_ + '/' + date[k]
            # List all categories of that specific date scan
            branch = os.listdir(date_)
            branch.sort()
            # only use first patient scan
            # Skip if encounter non-folder objects
            if branch[0].startswith('.'):
                continue
            else:
                branch_ = date_ + '/' + branch[0]
                # read files
                images_ = [pydicom.read_file(branch_ + '/' + s) for s in os.listdir(branch_) if s.endswith('.dcm')]
                images_.sort(key=lambda x: int(x.InstanceNumber))

    return images_, k_


def init_abnormal(input_f_, pat_id_):
    patients_list = os.listdir(input_f_)
    # might be multiple abnormalities in one image, they have different folders
    folders = []
    for name_ in patients_list:
        if name_.startswith(pat_id_):
            folders.append(name_)

    # if there are multiple folders, we need to make a list of segmentations and cropped images
    segmentations = []
    crops = []
    df_all = pd.DataFrame(
        {'patient_id': [], 'view_mode': [], 'orientation': [], 'abnormality_id': [], 'pathology_name': [],
         'pathology': [], 'merged_name': []})

    # Get segmentations of abnormalities
    for a in range(0, len(folders)):
        # Skip if non-folder objects are encountered
        if folders[a].startswith('.'):
            continue
        else:
            folder = input_f_ + folders[a]
            branch = os.listdir(folder)
            branch.sort()
            # Skip if encounter non-folder objects
            if branch[0].startswith('.'):
                continue
            else:
                branch_ = folder +'/' + branch[0]
                branch2_ = os.listdir(branch_)
                branch2_.sort()
                # Skip if encounter non-folder objects
                if branch2_[0].startswith('.'):
                    continue
                else:
                    # get all segmentation abnormalities
                    segmens_ = [pydicom.read_file(branch_ + '/' + branch2_[0] + '/' + s) for s in
                                os.listdir(branch_ + '/' + branch2_[0]) if s.endswith('.dcm')]
                    if len(segmens_) == 1:  # cropped image or segmentations is missing, so we skipt it
                        print('Missing image crop or segmentation of ', pat_id_)
                        continue
                    # the first entry is the cropped image, the second is the actual segmentation label (one we need)
                    # however, this is not consistent throughout the dataset... Therefore we need to check which one is
                    # the segmentation and crop
                    if segmens_[0].SeriesDescription == 'cropped images':
                        segmentations.append(segmens_[1])
                        crops.append(segmens_[0])
                    else:
                        segmentations.append(segmens_[0])
                        crops.append(segmens_[1])

                    # get malignancy through file path
                    sections = folders[a].split("_")
                    side = sections[3]
                    pat_name = sections[1] + '_' + sections[2]
                    view = sections[4]
                    ab_id = int(sections[-1])
                    # extract malignancy info from .csv
                    df = pd.read_csv(input_f_[:-14] + 'mass_case_description_test_set.csv')     # \CHANGE
                    df_id = df[(df['patient_id'] == pat_name) & (df['image view'] == view) &
                               (df['abnormality id'] == ab_id)].reset_index()
                    malignancy = df_id['pathology'][0]
                    if malignancy == 'MALIGNANT':
                        benign = 1
                    else:
                        benign = 0

                    if ab_id > 1:
                        df_info = pd.DataFrame(
                            {'patient_id': pat_name, 'view_mode': view, 'orientation': side,
                             'abnormality_id': ab_id,
                             'pathology_name': malignancy, 'pathology': benign,
                             'merged_name': [pat_name + '_' + view + '-' + side + '_' + str(ab_id)]})
                        df_all = pd.concat([df_all, df_info], ignore_index=True)
                    else:
                        df_info = pd.DataFrame(
                            {'patient_id': pat_name, 'view_mode': view, 'orientation': side,
                             'abnormality_id': ab_id,
                             'pathology_name': malignancy, 'pathology': benign,
                             'merged_name': [pat_name + '_' + view + '-' + side]})
                        df_all = pd.concat([df_all, df_info], ignore_index=True)

    return segmentations, crops, df_all


def preprocess_image(image_, img_size=(512, 512), use_clahe=False):
    # created with help of: https://github.com/CleonWong/Can-You-Find-The-Tumour/tree/master
    # 1: Get pixel array
    im_ = image_.pixel_array

    # 2: remove borders --> white lines around edges, this removes 1% left/right and 4% up/down
    nr, nc = im_.shape
    crop_im_ = im_[int(nr * 0.04):int(nr * (1 - 0.04)), int(nc * 0.01):int(nc * (1 - 0.01))]

    # 3: normalize image
    norm_im_ = (crop_im_ - crop_im_.min()) / (crop_im_.max() - crop_im_.min())

    # 4: remove in image labels + noise
    neat_im_, mask_ = remove_noise_artifacts(norm_im_)

    # 5: horizontal flip to make sure breasts are all facing the same direction; important for next steps
    # to check if the image needs flipping, we check on which side the mask has the highest values (1s)
    flip_im_, flip_mask, flip = flip_image(neat_im_, mask_)

    if use_clahe:
        # 5.5: enhance contrast  --> Do we want this? Might be easier for the model, but if radiologists don't use this
        #                            we shouldn't use this imo
        next_im = clahe(flip_im_)
    else:
        next_im = flip_im_

    # 6: pad images into squares (pad to the right hand side, which should be all zeros at this point)
    pad_im_ = pad2square(next_im)
    pad_mask_ = pad2square(flip_mask)

    # 7: downsample to 512x512 to make easily compatible with our current network?
    final_img = transform.resize(pad_im_, img_size, anti_aliasing=True)
    final_mask = transform.resize(pad_mask_, img_size, anti_aliasing=True)

    return final_img, flip, final_mask


def process_segmentation(segs_, flipped, data_frame, seg_size=(512, 512)):
    # process segmenations to get the same format as images and add multiple segmentations together
    segs_process = []
    xy_df_all = pd.DataFrame({'x_position': [], 'y_position': []})  # to add xy coordinates of abnormalities
    # 1: process segmenations to get same shape and orientation
    for n in range(0, len(segs_)):
        # 1.1: get pixel array
        seg = segs_[n].pixel_array

        # 1.2: crop borders
        nr, nc = seg.shape
        crop_seg_ = seg[int(nr * 0.04):int(nr * (1 - 0.04)), int(nc * 0.01):int(nc * (1 - 0.01))]

        # 1.3: horizontal flip if necessary
        if flipped:
            new_seg = np.fliplr(crop_seg_)  # flip segmenation
        else:
            new_seg = crop_seg_

        # 1.4: pad image
        pad_seg = pad2square(new_seg)

        # 1.5: normalize to 0 and 1
        norm_im_ = (pad_seg - pad_seg.min()) / (pad_seg.max() - pad_seg.min())

        # 1.6: add to list
        segs_process.append(norm_im_)

        # 1.7 add x and y coordinates of abnormality to df
        contour_im = transform.resize(norm_im_, seg_size, anti_aliasing=True)
        get_contours = np.zeros(seg_size, dtype="uint8")
        get_contours[contour_im != 0] = 1
        contours, hierarchy = cv2.findContours(get_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in range(len(contours)):  # for every contour/nodule found
            if contours[c].shape[0] < 10:
                continue
            else:
                x, y, w, h = cv2.boundingRect(contours[c])
                #  turn into yolo coordinates (normalized xywh format)
                x_center = (x + (w / 2))
                y_center = (y + (h / 2))
                xy_df = pd.DataFrame({'x_position': [x_center], 'y_position': [y_center]})
                xy_df_all = pd.concat([xy_df_all, xy_df], ignore_index=True)

    # 2: add segmenations together if multiple
    if len(segs_) > 1:
        all_seg = np.zeros(segs_process[0].shape, np.uint8)
        for s in range(0, len(segs_)):
            all_seg[segs_process[s] == 1] = 1
    else:
        all_seg = segs_process[0]

    # 3: downsample to 512x512 to make easily compatible with our current network?
    final_seg = transform.resize(all_seg, seg_size, anti_aliasing=True)
    # 4: add xy coordinates to total df
    df_final = pd.concat([data_frame, xy_df_all], axis=1, ignore_index=True)

    return final_seg, df_final


def create_semantic_map(breast, mass):
    sem_map = np.zeros(breast.shape, np.uint8)
    sem_map[breast != 0] = 1
    sem_map[mass != 0] = 2

    return sem_map


def get_cropped_images(image_crops, to_flip, crop_size=(64, 64)):
    new_crops = []
    # normalize
    for c in range(0, len(image_crops)):
        crop = image_crops[c].pixel_array
        norm_c = (crop - crop.min()) / (crop.max() - crop.min())
        # possibly flip
        if to_flip:
            norm_c = np.fliplr(norm_c)
        # resize
        final_crop = transform.resize(norm_c, crop_size, anti_aliasing=True)
        new_crops.append(final_crop)

    return new_crops


def save_info(img, smap, cropped, data_f, patient_id_, output_folder_, my_dpi=600, use_clahe=False):
    # first save df to csv
    file_name = output_folder_ + '/20230714_CBIS-DDSM_Lesion_Properties_Testset.csv'    # CHANGE
    data_f.to_csv(file_name, mode='a', index=False, header=False)
    # check if three folders exist and save image, label, put to that folder
    # create name:
    parts = patient_id_.split('_')
    file = parts[1] + '_' + parts[2] + '_' + parts[-1] + '_' + parts[-2] + '.png'

    # 1: save image
    if use_clahe:
        output_folder = output_folder_ + '/images_clahe/'
    else:
        output_folder = output_folder_ + '/images/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # 665 or my_dpi might need to be adjusted depending on your screen resolution
    plt.figure(figsize=(665 / my_dpi, 665 / my_dpi), dpi=my_dpi)
    plt.imshow(img, cmap=plt.cm.gray, vmin=img.min(), vmax=img.max())
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, file), bbox_inches='tight', pad_inches=0, dpi=my_dpi)
    plt.close()

    # 2: save visible label map
    output_folder = output_folder_ + '/put/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # 665 or my_dpi might need to be adjusted depending on your screen resolution
    plt.figure(figsize=(665 / my_dpi, 665 / my_dpi), dpi=my_dpi)
    plt.imshow(smap)
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, file), bbox_inches='tight', pad_inches=0, dpi=my_dpi)
    plt.close()

    # 3: save label map used for training network
    output_folder = output_folder_ + '/label/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    label = im.fromarray(smap)
    label = label.convert("L")
    label.save(output_folder + file)

    # 4: save cropped image
    idx = 1
    for cr in cropped:
        output_folder = output_folder_ + '/crops/'
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        # 84 or my_dpi might need to be adjusted depending on your screen resolution to get to 64x64
        plt.figure(figsize=(84 / my_dpi, 84 / my_dpi), dpi=my_dpi)
        plt.imshow(cr, cmap=plt.cm.gray, vmin=cr.min(), vmax=cr.max())
        plt.axis('off')
        if idx > 1:
            new_file = parts[1] + '_' + parts[2] + '_' + parts[-1] + '_' + parts[-2] + '_' + str(idx) + '.png'
            plt.savefig(os.path.join(output_folder, new_file), bbox_inches='tight', pad_inches=0, dpi=my_dpi)
        else:
            plt.savefig(os.path.join(output_folder, file), bbox_inches='tight', pad_inches=0, dpi=my_dpi)
        plt.close()
        idx += 1


def create_labels(patient_id_, input_folder_, output_folder_):

    # 1: Initialize DICOM image data
    images_, id_ = init_(input_folder_ + 'files/CBIS-DDSM/', patient_id_)
    # preprocess image
    image_, flip, mask_b = preprocess_image(images_[0], use_clahe=True)

    # 2: Initialize DICOM segmentation data of pathology (masses) and get cropped images for classification + save
    #    information of malignancy to csv
    segmentations_, crop_masses, df = init_abnormal(input_folder_ + 'ROI/CBIS-DDSM/', patient_id_)
    if not segmentations_:
        return
    # 2.1: process segmentation to get same shape and orientation as image and add multiple segmentations together
    segment_, df2 = process_segmentation(segmentations_, flip, df)
    # 2.2: process cropped images
    crops = get_cropped_images(crop_masses, flip)

    # 3: Further create semantic label map
    sem_map = create_semantic_map(mask_b, segment_)

    # 6: save image, semantic map (visual), semantic map (for network) and save cropped image
    save_info(image_, sem_map, crops, df2, patient_id_, output_folder_, use_clahe=True)


#################################
# Parameters and initialization #
#################################
INPUT_FOLDER = 'D:/Details/datasets/BreastMam/Testset/'  # LIDC-IDRI folder
OUTPUT_FOLDER = 'D:/Details/datasets/BreastMam/Testset/'  # Output folder.

# Determine patient IDs to run
ids = os.listdir(INPUT_FOLDER + 'files/CBIS-DDSM/')

run_list = []
for names in ids:
    if names.endswith("MLO"):       # we're only using images in the Mediolateral Oblique view
        run_list.append(names)

#run_list2 = run_list[600:]
# Preprocess and Generation of semantic labels and save lung nodule annotations/locations
for patient_id in run_list:
    print("Processing patient: ", patient_id)
    create_labels(patient_id, INPUT_FOLDER, OUTPUT_FOLDER)
