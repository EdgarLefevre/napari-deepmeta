import os

import cc3d
import cv2
import numpy as np
import skimage.exposure as exposure
import skimage.measure as measure
import skimage.transform as transform
import torch
from scipy import ndimage

import napari_deepmeta.models as models

################
# SEGMENTATION #
################


def process_img(mouse, model):
    output_stack = []
    for slice in mouse:
        slice = slice.reshape(1, 1, 128, 128)
        slice = torch.from_numpy(slice).float()
        output = model(slice)
        output = output.max(1).indices
        output_stack.append(output.cpu().detach().numpy())
    return output_stack


def contrast_and_reshape(mouse, size=128):
    # if len(mouse.shape) > 2:
    data = []
    for i in np.arange(mouse.shape[0]):
        img_adapteq = exposure.equalize_adapthist(mouse[i], clip_limit=0.03)
        data.append(img_adapteq)
    return np.array(data).reshape(-1, 1, size, size)
    # else:
    #     img_adapteq = exposure.equalize_adapthist(mouse, clip_limit=0.03)
    #     return np.array(img_adapteq).reshape(1, size, size)


def segment_stack(img):
    device = "cuda" if torch.cuda.is_available() else "cpu"  # marche pas
    model = models.Unet3plus()  # .to(device)
    model.load_state_dict(
        torch.load(
            os.path.dirname(os.path.realpath(__file__))
            + "/resources/model.pth",
            map_location=torch.device(device),
        )
    )
    model.eval()
    return model(img)  # .to(device))


def prepare_mouse(img, contrast):
    img = transform.resize(img, (len(img), 128, 128), anti_aliasing=True)
    img = img / np.amax(img)
    if contrast:
        return torch.Tensor(contrast_and_reshape(img))
    else:
        return torch.Tensor(img.reshape(-1, 1, 128, 128))


##################
# POSTPROCESSING #
##################


def remove_blobs(mask, min_size=10):
    mask = mask.reshape(128, 128).astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
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
    kernel1 = np.ones((k1, k1), np.uint8)
    kernel2 = np.ones((k2, k2), np.uint8)
    img_dilation = cv2.dilate(img, kernel1, iterations=1)
    return cv2.erode(img_dilation, kernel2, iterations=1)


def laplace(img_stack, mask_list):
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
            mask_list[(len(mask_list) - 1) - i] = np.zeros((128, 128))
        else:
            break
    return np.array(mask_list)


def sanity_check(mask_list):
    mask_list[0] = np.zeros((128, 128))
    mask_list[-1] = np.zeros((128, 128))
    for i in range(1, len(mask_list) - 1):
        if mask_list[i].sum() > 15:
            if mask_list[i - 1].sum() < 15 and mask_list[i + 1].sum() < 15:
                mask_list[i] = np.zeros((128, 128))
        else:
            mask_list[i] = np.zeros((128, 128))
    return mask_list


def postprocess(inputs, masks):
    inputs = inputs.detach().numpy()
    masks = laplace(inputs, masks)
    lungs_masks = np.array([mask > 0.5 for mask in masks])
    metas_masks = np.array([mask > 1.5 for mask in masks])
    lungs_masks = sanity_check(lungs_masks)
    lungs_masks = [remove_blobs(mask, 10) for mask in lungs_masks]
    lungs_masks = (
        np.array([dilate_and_erode(mask, 3, 3) for mask in lungs_masks]) / 255
    )
    metas_masks = [remove_blobs(mask, 3) for mask in metas_masks]
    metas_masks = (
        np.array([dilate_and_erode(mask, 3, 3) for mask in metas_masks]) / 255
    )
    return np.where((metas_masks == 1), 2, lungs_masks)


###########
# Drawing #
###########


def add_z(arr, z):
    return [np.insert(couple, 0, z) for couple in arr]


def border_detected(k, seg):
    return measure.find_contours(seg[k], 0.8)


def mask_to_plottable_3D(masks):
    plottable_list = []
    for i in range(len(masks)):
        contours = border_detected(i, masks)
        for contour in contours:
            contour = add_z(contour, i)
            if len(contour) > 0:
                plottable_list.append(contour)
    return plottable_list


def get_meta_nb(masks):
    _, N = cc3d.connected_components(
        np.array(masks), return_N=True, connectivity=18
    )
    return N
