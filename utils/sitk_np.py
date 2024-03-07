
import numpy as np
import SimpleITK as sitk

def sitk_to_npimage(image:sitk.Image, transpose_axis = True):
    image_array = sitk.GetArrayFromImage(image).astype(np.float32)
    if transpose_axis:
        image_array = np.transpose(image_array, (2, 1, 0))
    return image_array

def npimage_to_sitk(image:np.array, transpose_axis = True):
    if transpose_axis:
        image = np.transpose(image, (2, 1, 0))
    return sitk.GetImageFromArray(image)

def sitk_to_np_no_copy(image_sitk):
    return sitk.GetArrayViewFromImage(image_sitk)

def sitk_to_np(image_sitk, type=None):
    if type is None:
        return sitk.GetArrayFromImage(image_sitk)
    else:
        return sitk.GetArrayViewFromImage(image_sitk).astype(type)


def np_to_sitk(image_np, type=None, is_vector=False):
    if type is None:
        return sitk.GetImageFromArray(image_np, is_vector)
    else:
        return sitk.GetImageFromArray(image_np.astype(type), is_vector)


def sitk_list_to_np(image_list_sitk, type=None, axis=0):
    image_list_np = []
    for image_sitk in image_list_sitk:
        image_list_np.append(sitk_to_np_no_copy(image_sitk))
    np_image = np.stack(image_list_np, axis=axis)
    if type is not None:
        np_image = np_image.astype(type)
    return np_image
