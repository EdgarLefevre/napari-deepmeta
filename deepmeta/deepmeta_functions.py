"""
Deepmeta functions
====================
In this section you will find all information about functions relative to deepmeta's original code.
"""

import numpy as np
from pathlib import Path
from appdirs import user_config_dir
from configparser import ConfigParser
from scipy import ndimage
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import cv2
import os
import skimage.measure as measure
import skimage.exposure as exposure


def predict_seg(dataset, path_model_seg, thresh=0.5):
    """
    Run inference.

    :param dataset: Imgs to segment
    :type dataset: np.array
    :param path_model_seg: Path of model
    :type path_model_seg: str
    :param thresh: Threshold value to binarize output mask [0,1]
    :type thresh: float
    :return: List of output binary masks
    :rtype: np.array
    """
    if "weighted" not in path_model_seg:
        model_seg = keras.models.load_model(
            path_model_seg,
        )
    else:
        model_seg = keras.models.load_model(
            path_model_seg,
            custom_objects={
                "weighted_cross_entropy": weighted_cross_entropy
            },
        )
    res = model_seg.predict(dataset)
    return (res > thresh).astype(np.uint8).reshape(len(dataset), 128, 128, 1)


def border_detected(k, seg):
    """
    Get borders from mask.

    :param k: Index of the image
    :type k: int
    :param seg: Mask
    :type seg: np.array
    """
    cell_contours = measure.find_contours(seg[k], 0.8)
    return cell_contours


def weighted_cross_entropy(y_true, y_pred):
    """
    Weighted cross entropy loss

    :param y_true: Ground truth
    :param y_pred: Prediction
    :return: Loss value between y_true and y_pred
    """
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=3)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except Exception:
        # test purpose
        seg = tf.zeros((1,128,128,1))
        weight = tf.ones((1, 128, 128, 1))
    epsilon = tf.convert_to_tensor(10e-8, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    zeros = tf.zeros_like(y_pred, dtype=y_pred.dtype)  # array_ops
    cond = y_pred >= zeros
    relu_logits = tf.where(cond, y_pred, zeros)
    neg_abs_logits = tf.where(cond, -y_pred, y_pred)
    entropy = tf.math.add(
        relu_logits - y_pred * seg,
        tf.math.log1p(tf.math.exp(neg_abs_logits)),
        name=None,
    )
    return K.mean(tf.multiply(weight, entropy), axis=-1)


def postprocess_loop(seg, cfg):
    """
    Run all post process actions on network output

    :param seg: Network output (binarized)
    :type seg: np.array
    :param cfg: Config object
    :type cfg: ConfigParser
    :return: Seg postprocessed
    :rtype: np.array
    """
    res = []
    for elt in seg:
        blobed = remove_blobs(elt)
        eroded = dilate_and_erode(blobed,
                                  int(cfg["Deepmeta"]["Kernel1_size_lungs"]),
                                  int(cfg["Deepmeta"]["Kernel2_size_lungs"])
                                  )
        res.append(eroded / 255)
    return np.array(res)


def postprocess_meta(seg, k1, k2):
    """
    Run postprocess loop for metas

    :param seg: Network output
    :type seg: np.array
    :param k1: Size of dilatation kernel
    :type k1: int
    :param k2: Size of erosion kernel
    :type k2: int
    :return: Postprocessed seg
    :rtype: np.array
    """
    res = []
    for elt in seg:
        elt = remove_blobs(elt, min_size=3)
        res.append(dilate_and_erode(elt, k1=k1, k2=k2))
    return np.array(res)


def remove_blobs(img, min_size=10):
    """
    Remove small blobs in mask.

    :param min_size: Min size of blob to remove (pixels)
    :type min_size: int
    :param img: Mask
    :type img: np.array
    :return: Mask without blobs
    :rtype: np.array
    """
    img = img.astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=8
    )
    sizes = stats[1:, -1]
    nb_components = nb_components - 1  # remove background
    img2 = np.zeros(output.shape)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2


def dilate_and_erode(img, k1=3, k2=3):
    """
    Dilate and erode a mask.

    :param img: Mask
    :type img: np.array
    :param k1: Size of dilatation kernel
    :type k1: int
    :param k2: Size of erosion kernel
    :type k2: int
    :return: Mask processed
    :rtype: np.array
    """
    kernel1 = np.ones((k1, k1), np.uint8)
    kernel2 = np.ones((k2, k2), np.uint8)

    img_dilation = cv2.dilate(img, kernel1, iterations=1)
    img_erosion2 = cv2.erode(img_dilation, kernel2, iterations=1)
    return img_erosion2


def add_z(arr, z):
    """
    Add z dimension to contours lines.

    :param arr: Contour line list
    :type arr: List
    :param z: Z index to add
    :type z: int
    :return: Arr with a z dim added
    :rtype: List
    """
    final_arr = []
    for couple in arr:
        final_arr.append(np.insert(couple, 0, z))
    return np.array(final_arr)


def from_mask_to_non_plottable_list(masks):
    """
    Create a list of plottable elements (borders) from a mask list.

    :param masks: List of np arrays containing segmentation results
    :type masks: nd.array
    :return: A list of plottable elements
    :rtype: List
    """
    non_plottable_list = []
    for i in range(len(masks)):
        plottable = []
        contours = border_detected(i, masks)
        for contour in contours:
            contour = add_z(contour, i)
            plottable.append(contour)
        if len(plottable) > 0:
            non_plottable_list.append(plottable)
    return non_plottable_list


def seg_lungs_(image, cfg):
    """
    Segment and postprocess lungs.

    :param image: Image values in [0,1]
    :type image: np array
    :param cfg: Config parser object
    :type cfg: ConfigParser
    :return: Masks list
    :rtype: List
    """
    path_model_seg = cfg["Deepmeta"]["path_model_lungs"]
    masks = predict_seg(image, path_model_seg).reshape(len(image), 128, 128)
    masks = laplace(image, masks)
    masks = lungs_sanity_check(masks)
    masks = postprocess_loop(masks, cfg)
    return masks


def seg_lungs(image, cfg):
    """
    Segment lungs.

    :param image: Image values in [0,1]
    :type image: np array
    :param cfg: Config parser object
    :type cfg: ConfigParser
    :return: A list of contours lines, list of volumes
    :rtype: List, List
    """
    masks = seg_lungs_(image, cfg)
    return from_mask_to_non_plottable_list(masks), get_volumes(masks, float(cfg["Deepmeta"]["volume"]))


def contrast_and_reshape(mouse, size=128):
    """
    Enhance image contrast.

    :param mouse: Slices of the mouse we want to segment
    :type mouse: np.array
    :param size: Size of the images (we assume images are squares)
    :type size: int
    :return: Images list with readjusted contrast
    :rtype: np.array

    .. warning:
       If the contrast of the mouse should not be readjusted, the network will fail prediction.
       Same if the image should be contrasted and you do not run it.
    """
    if len(mouse.shape) > 2:
        data = []
        for i in np.arange(mouse.shape[0]):
            img_adapteq = exposure.equalize_adapthist(
                mouse[i], clip_limit=0.03
            )  # clip_limit=0.03 de base
            data.append(img_adapteq)
        data = np.array(data).reshape(-1, size, size, 1)
        return data
    else:
        img_adapteq = exposure.equalize_adapthist(mouse, clip_limit=0.03)
        img = np.array(img_adapteq).reshape(size, size, 1)
        return img


def seg_metas(image, cfg):
    """
    Segment metastasis.

    :param image: Image values in [0,1]
    :type image: np.array
    :return: A list of contours lines, list of volumes
    :rtype: List, List
    """
    lungs_masks = seg_lungs_(image, cfg)
    path_model_seg = cfg["Deepmeta"]["path_model_metas"]
    masks = predict_seg(image, path_model_seg).reshape(len(image), 128, 128)
    masks = (lungs_masks * masks).reshape(len(image), 128, 128)
    masks = postprocess_meta(masks,
                             int(cfg["Deepmeta"]["Kernel1_size_metas"]),
                             int(cfg["Deepmeta"]["Kernel2_size_metas"])
                             )
    masks /= 255
    return from_mask_to_non_plottable_list(masks), get_volumes(masks, float(cfg["Deepmeta"]["volume"]))


def get_volumes(masks, vol):
    """
    Get each volumes (volume on slice).

    :param masks: Pred mask list
    :type masks: List
    :param vol: Volume of one voxel
    :type vol: float
    :return: A list of all volumes per slice
    :rtype: List
    """
    res = []
    for mask in masks:
        tmp = []
        labels = measure.label(mask, connectivity=1)
        for i in range(1, labels.max() + 1):
            tmp.append(
                (labels == i).sum() * vol
            )
        if len(tmp) > 0:
            res.append(tmp)
    return res



def load_config():
    """
    Function to parse config file, create default one in ~/.config/deepmeta/config.ini.

    :return: An object with all config file parsed
    :rtype: ConfigParser
    """
    cfg_loc = Path(user_config_dir(appname="deepmeta")) / "config.ini"
    if not cfg_loc.exists():
        cfg_loc.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_loc, mode="a+") as f:
            f.write(
                "[Deepmeta]\n"
                "volume = 0.0047\n"
                "path_model_lungs = " + os.path.dirname(os.path.realpath(__file__)) + "/resources/models/model_lungs_weighted.h5\n"
                "path_model_metas = " + os.path.dirname(os.path.realpath(__file__)) + "/resources/models/model_metas_weighted.h5\n"
                "Kernel1_size_lungs = 3\n"
                "Kernel2_size_lungs = 3\n"
                "Kernel1_size_metas = 3\n"
                "Kernel2_size_metas = 3\n"
                "color_lungs = red\n"
                "color_metas = blue\n"
            )
        print(f"Initialized new default config at {cfg_loc}.")
    cfg = ConfigParser()
    cfg.read(cfg_loc)
    return cfg


def laplace(img_stack, mask_list):
    """
    Remove false positives in lung segmentation. Apply a laplace of gaussian filter on slices, if the mean value of the
    result is <1 we remove the mask.

    .. note::
       We process only first and last slices (until we find a value >1). This ensure that we do not remove false
       negative slices.

    :param img_stack: Full image stack (dataset).
    :type img_stack: np.array
    :param mask_list: Full lung segmentation output
    :type mask_list: np.array
    :return: Updated mask list
    :rtype: np.array
    """
    img_stack2 = (img_stack * 255).astype(np.uint8)
    for i, img in enumerate(img_stack2):
        new_im = ndimage.gaussian_laplace(img, sigma=7)
        if np.mean(new_im) < 1:
            mask_list[i] = np.zeros((128, 128))
        else:
            break
    for i, img in enumerate(img_stack2[::-1]):
        new_im = ndimage.gaussian_laplace(img, sigma=7)
        if np.mean(new_im) < 1:
            mask_list[(len(mask_list)-1)-i] = np.zeros((128, 128))
        else:
            break
    return mask_list


def lungs_sanity_check(mask_list):
    """
    Check if there is some false positive. If mask < 15px -> mask is null.
    If i-1 and i+1 do not contain mask, i does not contains a mask either.

    :param mask_list: Lungs segmentation output
    :type mask_list: np.array
    :return: Checked segmentation output
    :rtype: np.array
    """
    mask_list[0] = np.zeros((128, 128))
    mask_list[-1] = np.zeros((128, 128))
    for i in range(1, len(mask_list)-1):
        if mask_list[i].sum() > 15:
            if mask_list[i-1].sum() < 15 and mask_list[i+1].sum() < 15:
                mask_list[i] = np.zeros((128, 128))
        else:
            mask_list[i] = np.zeros((128, 128))
    return mask_list