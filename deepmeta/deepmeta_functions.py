import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import cv2
import skimage.measure as measure
import skimage.exposure as exposure


def predict_seg(dataset, path_model_seg, tresh=0.5):
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
    return (res > tresh).astype(np.uint8).reshape(len(dataset), 128, 128, 1)


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
    -- Fonction de coût pondéré --
    :param y_true: vrai valeur de y (label)
    :param y_pred: valeur prédite de y par le modèle
    :return: valeur de la fonction de cout d'entropie croisée pondérée
    """
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=3)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except Exception:
        pass

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


def postprocess_loop(seg):
    res = []
    for elt in seg:
        blobed = remove_blobs(elt)
        eroded = dilate_and_erode(blobed)
        res.append(eroded / 255)
    return np.array(res)


def postprocess_meta(seg, k1=3, k2=3):
    res = []
    for elt in seg:
        res.append(dilate_and_erode(elt, k1=k1, k2=k2))  # try with 5x5
    return np.array(res)


def remove_blobs(img):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=8
    )
    sizes = stats[1:, -1]
    nb_components = nb_components - 1  # remove background
    min_size = 10
    img2 = np.zeros(output.shape)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2


def dilate_and_erode(img, k1=3, k2=3):
    kernel1 = np.ones((k1, k1), np.uint8)
    kernel2 = np.ones((k2, k2), np.uint8)

    img_dilation = cv2.dilate(img, kernel1, iterations=1)
    img_erosion2 = cv2.erode(img_dilation, kernel2, iterations=1)
    return img_erosion2


def add_z(arr, z):
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


def seg_lungs_(image):
    path_model_seg = "/home/edgar/Documents/Projects/DeepMeta/data/saved_models/Poumons/best_seg_model_weighted.h5"
    masks = predict_seg(image, path_model_seg).reshape(128, 128, 128)
    masks = postprocess_loop(masks)
    return masks


def seg_lungs(image):
    """

    :param image: Image values in [0,1]
    :type image: np array
    :return:
    :rtype:
    """
    masks = seg_lungs_(image)
    return from_mask_to_non_plottable_list(masks), get_volumes(masks)

def contrast_and_reshape(souris, size=128):
    """
    For some mice, we need to readjust the contrast.

    :param souris: Slices of the mouse we want to segment
    :type souris: np.array
    :param size: Size of the images (we assume images are squares)
    :type size: int
    :return: Images list with readjusted contrast
    :rtype: np.array

    .. warning:
       If the contrast of the mouse should not be readjusted, the network will fail prediction.
       Same if the image should be contrasted and you do not run it.
    """
    if len(souris.shape) > 2:
        data = []
        for i in np.arange(souris.shape[0]):
            img_adapteq = exposure.equalize_adapthist(
                souris[i], clip_limit=0.03
            )  # clip_limit=0.03 de base
            data.append(img_adapteq)
        data = np.array(data).reshape(-1, size, size, 1)
        return data
    else:
        img_adapteq = exposure.equalize_adapthist(souris, clip_limit=0.03)
        img = np.array(img_adapteq).reshape(size, size, 1)
        return img


def seg_metas(image):
    """

    :param image: Image values in [0,1]
    :type image: np array
    :return:
    :rtype:
    """
    lungs_masks = seg_lungs_(image)
    path_model_seg = "/home/edgar/Documents/Projects/DeepMeta/data/saved_models/Metastases/best_seg_weighted.h5"
    masks = predict_seg(image, path_model_seg).reshape(128, 128, 128)
    masks = (lungs_masks * masks).reshape(128, 128, 128)
    masks = postprocess_meta(masks)
    return from_mask_to_non_plottable_list(masks), get_volumes(masks)


def get_volumes(masks, vol=0.0047):
    """
    Get each volumes (volume on slice).
    :param masks:
    :type masks:
    :param vol:
    :type vol:
    :return:
    :rtype:
    """
    res = []
    for mask in masks:
        tmp = []
        labels = measure.label(mask, connectivity=1)
        for i in range(1, labels.max()+1):
            tmp.append(
                (labels == i).sum() * vol
            )
        if len(tmp) > 0:
            res.append(tmp)
    return res
