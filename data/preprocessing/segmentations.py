import numpy as np
import scipy
from skimage import measure, morphology
from sklearn.cluster import KMeans


def semantic_nonROI(image_, range1_, range2_, range3_, range4_):
    # Hard copy the image
    clip_ = np.copy(image_)
    clip1_ = np.copy(image_)
    clip2_ = np.copy(image_)

    # Thresholding
    clip_[image_ < range1_] = 0
    clip_[image_ > range4_] = 0
    clip1_[image_ < range2_] = 0
    clip1_[image_ > range3_] = 0
    clip2_[image_ < range3_] = 0
    clip2_[image_ > range4_] = 0

    # Create Boolean Mask for each class
    mask_ = scipy.ndimage.morphology.binary_fill_holes(clip_)
    mask1_ = scipy.ndimage.morphology.binary_fill_holes(clip1_)
    mask2_ = scipy.ndimage.morphology.binary_fill_holes(clip2_)

    # Create semantic label map
    semantic_map_ = np.zeros((image_.shape[0], image_.shape[0]))
    semantic_map_[mask_ == True] = 1
    semantic_map_[mask1_ == True] = 2
    semantic_map_[mask2_ == True] = 3

    return semantic_map_, image_


def threshold_lung_parts(image_):
    # Code sourced from https://www.kaggle.com/c/data-science-bowl-2017#tutorial ###

    middle = image_[100:400, 100:400]
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img_ = np.where(image_ < threshold, 1.0, 0.0)

    return thresh_img_


def depict_contour(image_, param_, param2_):
    # param_: area of depicted contour, using as lower bound threshold; param2_ for higher bound threshold #

    # create binary mask by KMeans
    thresh_img = threshold_lung_parts(image_)
    x = measure.find_contours(thresh_img, 0.5)
    contours_ = []

    for c in range(0, len(x)):
        try:
            hull_ = scipy.spatial.ConvexHull(x[c])
            if param_ < hull_.volume < param2_:
                # if contour is closed
                if x[c][0][0] == x[c][len(x[c]) - 1][0] and x[c][0][1] == x[c][len(x[c]) - 1][1]:
                    contours_.append(x[c])
        except:
            continue

    return contours_


def depict_contour_area(image_):
    # NOT USED AT THIS POINT; def depict_contour is used instead ##
    thresh_img = threshold_lung_parts(image_)
    x = measure.find_contours(thresh_img, 0.5)
    contours_ = []
    area_ = []

    for i in range(0, len(x)):
        try:
            hull_ = scipy.spatial.ConvexHull(x[i])
            if hull_.volume >= 2:
                contours_.append(x[i])
                # print(hull_.volume)
                a = hull_.volume
                area_.append(int(a))
        except:
            continue

    return area_


def kmeans_lung(image_):
    # Code sourced from https://www.kaggle.com/c/data-science-bowl-2017#tutorial ###

    thresh_img = threshold_lung_parts(image_)
    eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
    dilation = morphology.dilation(eroded, np.ones([5, 5]))
    labels = measure.label(dilation)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        bb = prop.bbox
        if bb[2] - bb[0] < 475 and bb[3] - bb[1] < 475 and bb[0] > 40 and bb[2] < 472:
            good_labels.append(prop.label)
    lung = np.ndarray([image_.shape[0], image_.shape[0]], dtype=np.int8)
    lung[:] = 0
    for N in good_labels:
        lung = lung + np.where(labels == N, 1, 0)
    lung = morphology.dilation(lung, np.ones([3, 3]))  # one last dilation

    return lung


def contour_to_mask(num1_, num2_, contours_, image_, kmeans_):
    lung_mask_ = np.zeros((image_.shape[0], image_.shape[0]))
    for u in range(num1_, num2_):
        lung_ = np.zeros((image_.shape[0], image_.shape[0]))
        lung_[np.int_(contours_[u][:, 0]), np.int_(
            contours_[u][:, 1])] = 1  # arrays used as indices must be of integer (or boolean) type
        mask_ = scipy.ndimage.morphology.binary_fill_holes(lung_)
        lung_mask_ = lung_mask_ + mask_
    semantic_map_ = np.zeros((image_.shape[0], image_.shape[0]))
    semantic_map_[lung_mask_ == True] = 1
    if np.count_nonzero(semantic_map_) != 0:
        result_ = 1 - scipy.spatial.distance.cosine(kmeans_.flatten(), semantic_map_.flatten())
    else:
        result_ = 0

    return semantic_map_, result_


def depict_semantic_lung(image_, semantic_map_, param_, lower_bound_, param2_):
    # lower_bound_: similarity threshold between KMeans and FindCountour. Normally set to 0.7, but for diagnostic
    # set to 0 since true lung regions can be skipped.

    contours_ = depict_contour(image_, param_, param2_)
    if len(contours_) >= 1:
        km_ = kmeans_lung(image_)
        num1_ = 0
        semantic_list_ = []
        result_list_ = []

        # Iteratively combine contours and form the masks. The masks are compared with KMeans mask and given the
        # results of similarity.
        for con in range(num1_, len(contours_) + 1):
            semantic_, result_ = contour_to_mask(num1_, con, contours_, image_, km_)
            semantic_list_.append(semantic_)
            result_list_.append(result_)
            if con == len(contours_):
                semantic_, result_ = contour_to_mask(num1_ + 1, con, contours_, image_, km_)
                semantic_list_.append(semantic_)
                result_list_.append(result_)
                if len(contours_) >= 3:
                    semantic_, result_ = contour_to_mask(num1_ + 1, con - 1, contours_, image_, km_)
                    semantic_list_.append(semantic_)
                    result_list_.append(result_)
                    semantic_, result_ = contour_to_mask(num1_ + 2, con - 1, contours_, image_, km_)
                    semantic_list_.append(semantic_)
                    result_list_.append(result_)
                    if len(contours_) >= 4:
                        semantic_, result_ = contour_to_mask(num1_ + 1, con - 2, contours_, image_, km_)
                        semantic_list_.append(semantic_)
                        result_list_.append(result_)
                        semantic_, result_ = contour_to_mask(num1_ + 2, con - 2, contours_, image_, km_)
                        semantic_list_.append(semantic_)
                        result_list_.append(result_)
                        semantic_, result_ = contour_to_mask(num1_ + 1, con - 3, contours_, image_, km_)
                        semantic_list_.append(semantic_)
                        result_list_.append(result_)

        # If calculated similarity == nan, revise it to 0
        result_list_[result_list_ == 'nan'] = 0
        if max(result_list_) >= lower_bound_:
            maxim = result_list_.index(max(result_list_))
            # maxim = 2
            # for some special diagnostic cases, setting maxim = 1or2 solves the issues magically
            semantic_map_binary_ = semantic_list_[maxim]
            semantic_map_[semantic_map_binary_ == 1] = 4

            return semantic_map_
        else:
            return semantic_map_
    else:
        return semantic_map_

