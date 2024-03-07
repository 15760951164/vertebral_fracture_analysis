import json
import os
from glob import glob
import numpy as np
from scipy import ndimage
import random
import math
import SimpleITK as sitk
import shutil
from utils.sitk_np import sitk_to_npimage, npimage_to_sitk
from scipy.ndimage import center_of_mass, binary_erosion
from utils.sitk_image import resample_to_spacing
from utils.heatmap import generator_heatmaps_by_msk

def globalNormalization(x):
    import numpy as np 
    import sys
    from math import sqrt
    """
    Normalize the data by substract mean and then devided by std
    X(i) = x(i)-mean / sqrt(stdˆ2 + e)
    """

    mean = np.mean(x)
    std = np.std(x)

    epsilon = sys.float_info.epsilon

    x_vec = x.flatten().astype(np.float64)
    lengh = len(x_vec)
    for n in range(lengh):
        x_vec[n] = (x_vec[n] - mean)/(sqrt(std**2+epsilon))
    x_norm = np.resize(x_vec, x.shape)

    return x_norm

def vol_padding_ori(cube_size, vol, pad_value=-1024, front_after_pad_size=24):
    h, w, c = vol.shape
    padX = [front_after_pad_size, front_after_pad_size]
    padY = [front_after_pad_size, front_after_pad_size]
    padZ = [front_after_pad_size, front_after_pad_size]

    if h < cube_size[0]:
        padX[1] += cube_size[0] - h
    elif h % cube_size[0] != 0:
        padX[1] += cube_size[0] - (h % cube_size[0])

    if w < cube_size[1]:
        padY[1] += cube_size[1] - w
    elif w % cube_size[1] != 0:
        padY[1] += cube_size[1] - (w % cube_size[1])

    if c < cube_size[2]:
        padZ[1] += cube_size[2] - c
    elif c % cube_size[2] != 0:
        padZ[1] += cube_size[2] - (c % cube_size[2])

    vol = np.pad(
        vol,
        (tuple(padX), tuple(padY), tuple(padZ)),
        mode="constant",
        constant_values=pad_value,
    ).astype(np.float32)

    return vol, [padX, padY, padZ]

def vol_padding(vol,
                cube_size,
                pad_value=-1000):  # pad CT with -1000 (HU of air)
    h, w, c = vol.shape
    x, y, z = cube_size[:]

    pad_value = float(pad_value)

    if h < x:
        pad_width_1 = (x - h) // 2
        pad_width_2 = x - h - pad_width_1
        vol = np.pad(
            vol,
            [(pad_width_1, pad_width_2), (0, 0), (0, 0)],
            mode="constant",
            constant_values=pad_value,
        )

    if w < y:
        pad_width_1 = (y - w) // 2
        pad_width_2 = y - w - pad_width_1
        vol = np.pad(
            vol,
            [(0, 0), (pad_width_1, pad_width_2), (0, 0)],
            mode="constant",
            constant_values=pad_value,
        )

    if c < z:
        pad_width_1 = (z - c) // 2
        pad_width_2 = z - c - pad_width_1
        vol = np.pad(
            vol,
            [(0, 0), (0, 0), (pad_width_1, pad_width_2)],
            mode="constant",
            constant_values=pad_value,
        )

    assert (np.array(list(vol.shape)) >= np.array(list(cube_size))).any()

    return vol

def get_cubes(img, msk, cube_size, stride, norm=False):
    img, _ = vol_padding_ori(cube_size, img)
    msk, _ = vol_padding_ori(cube_size, msk, 0)

    if min(img.shape) < cube_size[0]:
        img = vol_padding(img, cube_size, pad_value=-1000)
        msk = vol_padding(msk, cube_size, pad_value=0)

    if norm:
        img = globalNormalization(img)

    print("labels in the mask: ", np.unique(msk))

    h, w, c = img.shape

    x, y, z = cube_size[:]

    img_cubes = []
    msk_cubes = []

    for i in range(0, h, stride):
        for j in range(0, w, stride):
            for k in range(0, c, stride):
                x_start, x_end = i, i + cube_size[0]
                y_start, y_end = j, j + cube_size[1]
                z_start, z_end = k, k + cube_size[2]

                if x_end > h or y_end > w or z_end > c:
                    continue

                img_cube = img[x_start:x_end, y_start:y_end, z_start:z_end]
                msk_cube = msk[x_start:x_end, y_start:y_end, z_start:z_end]

                img_cubes.append(img_cube)
                msk_cubes.append(msk_cube)

    img_cubes = np.array(img_cubes, dtype=np.float32)
    msk_cubes = np.array(msk_cubes, dtype=np.float32)

    return img_cubes, msk_cubes

def get_roi_cubes(img, msk, cube_size, stride, norm=False):
    h, w, c = img.shape
    x, y, z = cube_size[:]

    min_x, min_y, min_z, max_x, max_y, max_z = get_roi_bbox(msk)

    roi_h = max_x - min_x
    roi_w = max_y - min_y
    roi_c = max_z - min_z

    # if not (roi_h >= x and roi_w >= y and roi_c >= z):
    #     min_x, min_y, min_z, max_x, max_y, max_z = positioning_roi(
    #         min_x, min_y, min_z, max_x, max_y, max_z, h, w, c, cube_size
    #     )

    img_roi = img[min_x:max_x, min_y:max_y, min_z:max_z]
    msk_roi = msk[min_x:max_x, min_y:max_y, min_z:max_z]

    if min(img_roi.shape) < 96:
        img_roi = vol_padding(img_roi, cube_size, pad_value=-1000)
        msk_roi = vol_padding(msk_roi, cube_size, pad_value=0)

    if norm:
        img_roi = globalNormalization(img_roi)

    h, w, c = img_roi.shape

    img_cubes = []
    msk_cubes = []

    for i in range(0, h - x + 1, stride):
        for j in range(0, w - y + 1, stride):
            for k in range(0, c - z + 1, stride):
                img_cube = img_roi[i:i + x, j:j + y, k:k + z]
                msk_cube = msk_roi[i:i + x, j:j + y, k:k + z]

                img_cubes.append(img_cube)
                msk_cubes.append(msk_cube)

    img_cubes = np.array(img_cubes, dtype=np.float32)
    msk_cubes = np.array(msk_cubes, dtype=np.float32)

    return img_cubes, msk_cubes


def get_roi_bbox(msk):
    x_array, y_array, z_array = np.where(msk > 0)

    min_x, max_x = min(x_array), max(x_array)
    min_y, max_y = min(y_array), max(y_array)
    min_z, max_z = min(z_array), max(z_array)

    return min_x, min_y, min_z, max_x, max_y, max_z


def positioning_roi(min_x, min_y, min_z, max_x, max_y, max_z, h, w, c,
                    cube_size):
    x, y, z = cube_size[:]
    roi_h = max_x - min_x
    roi_w = max_y - min_y
    roi_c = max_z - min_z

    min_x, max_x = positioning_roi_width(min_x, max_x, roi_h, x, h)
    min_y, max_y = positioning_roi_width(min_y, max_y, roi_w, y, w)
    min_z, max_z = positioning_roi_width(min_z, max_z, roi_c, z, c)

    return min_x, min_y, min_z, max_x, max_y, max_z


def positioning_roi_width(min_x, max_x, roi_h, cube_size, h):
    if roi_h < cube_size:
        ready_flag = True
        expand_width = cube_size - roi_h
        if min_x - expand_width // 2 >= 0:
            min_x = -expand_width // 2
        else:
            min_x = 0
            ready_flag = False
        if max_x + (expand_width - expand_width // 2) <= h and ready_flag:
            max_x += expand_width - expand_width // 2

        elif not ready_flag:
            if cube_size <= h:
                max_x = cube_size
        elif max_x + (expand_width - expand_width // 2) > h and ready_flag:
            max_x = h
        else:
            pass

    return min_x, max_x


def ROI_pad(anchor, vol_size, ROI_size):
    vol_x, vol_y, vol_z = vol_size[:]
    centor_x, centor_y, centor_z = anchor[:]
    rx, ry, rz = ROI_size[:]

    padX = [0, 0]
    padY = [0, 0]
    padZ = [0, 0]

    if centor_x - rx // 2 < 0:
        padX[0] = rx // 2 - centor_x
    if centor_x + rx // 2 > vol_x:
        padX[1] = centor_x + rx // 2 - vol_x

    if centor_y - ry // 2 < 0:
        padY[0] = ry // 2 - centor_y
    if centor_y + ry // 2 > vol_y:
        padY[1] = centor_y + ry // 2 - vol_y

    if centor_z - rz // 2 < 0:
        padZ[0] = rz // 2 - centor_z
    if centor_z + rz // 2 > vol_z:
        padZ[1] = centor_z + rz // 2 - vol_z

    return padX, padY, padZ


def get_ROI_img(vol, anchor):

    anchor = np.round(anchor).astype(np.int32)

    x_centor, y_centor, z_centor = anchor[:]

    ROI_size = (200, 200, 200)

    x, y, z = vol.shape

    padX, padY, padZ = ROI_pad(anchor, vol.shape, ROI_size)

    img_crop = vol[max(x_centor -
                       ROI_size[0] // 2, 0):min(x_centor +
                                                ROI_size[0] // 2, x),
                   max(y_centor -
                       ROI_size[1] // 2, 0):min(y_centor +
                                                ROI_size[1] // 2, y),
                   max(z_centor -
                       ROI_size[2] // 2, 0):min(z_centor +
                                                ROI_size[2] // 2, z), ]

    img_pad = np.pad(img_crop, [padX, padY, padZ],
                     mode="constant",
                     constant_values=vol.min())

    assert img_pad.shape == ROI_size

    return img_pad



def read_image_and_mask(ID, dataset_dir, spacing=[1.0]*3):

    vol_files = glob(os.path.join(dataset_dir, ID, "{}*.nii.gz").format(ID))

    assert len(vol_files) == 2, f"please check {ID} folder"

    for file in vol_files:
        if "_seg" in file:
            msk_file = file
        else:
            img_file = file

    img = sitk_to_npimage(resample_to_spacing(sitk.ReadImage(img_file), spacing))
    msk = sitk_to_npimage(resample_to_spacing(sitk.ReadImage(msk_file), spacing))

    return img, msk


def read_annotation(ID, dataset_dir):
    # read the labels and locations
    anno_file = glob(os.path.join(dataset_dir, ID, "*.json"))[0]
    
    with open(anno_file, 'r') as f:
        data = f.read()
        anno = json.loads(data)

    locs = []
    labels = []

    for i in range(len(anno)):
        x = int(anno[i]["X"])
        y = int(anno[i]["Y"])
        z = int(anno[i]["Z"])
        label = int(anno[i]["label"])

        locs.append([x, y, z])
        labels.append(label)

    locs = np.array(locs).astype(np.float32)
    labels = np.array(labels)

    annotations = {"locations": locs, "labels": labels}

    return annotations


def generate_idv_segmentor_datasets(ID, dataset_dir, save_folder):
    
    check_path(save_folder)

    img_save_dir = os.path.join(save_folder, "img")
    msk_save_dir = os.path.join(save_folder, "msk")

    check_path(img_save_dir)
    check_path(msk_save_dir)

    print(" ... processing ID: ", ID)

    pir_img, pir_msk = read_image_and_mask(ID,
                                           dataset_dir)

    anno = read_annotation(ID, dataset_dir)
    locations = anno["locations"]
    labels = anno["labels"]

    for i, (loc, label) in enumerate(zip(locations, labels)):
        gt_msk = (pir_msk == label).astype(np.int32)
        gt_img = np.copy(pir_img)

        img_roi = get_ROI_img(gt_img, loc)
        msk_roi = get_ROI_img(gt_msk, loc)
        msk_roi = np.round(msk_roi).astype(np.int32)

        if len(np.unique(msk_roi)) != 2:
            continue

        assert len(np.unique(msk_roi)) == 2
        assert img_roi.shape == msk_roi.shape

        rot_range = [np.random.randint(-35, 35) for _ in range(2)]
        rot_range.append(0)

        for rot_z in rot_range:
            axes = [(1, 0), (2, 1), (2, 0)]
            axis = axes[np.random.randint(len(axes))]
            shift_x = np.random.randint(-15, 15)
            shift_y = np.random.randint(-15, 15)
            shift_z = np.random.randint(-15, 15)

            if rot_z == 0:
                shift_x, shift_y, shift_z = 0, 0, 0

            img_rot = ndimage.rotate(img_roi,
                                     rot_z,
                                     axis,
                                     cval=img_roi.min(),
                                     reshape=False)

            msk_rot = ndimage.rotate(msk_roi,
                                     rot_z,
                                     axis,
                                     cval=0,
                                     reshape=False)
            msk_rot = np.round(msk_rot).astype(np.int32)

            img_rot_copy = np.copy(img_rot)
            msk_rot_copy = np.copy(msk_rot)

            if shift_x < 0:
                size_range_x = [0 - shift_x, 200]
            else:
                size_range_x = [0, 200 - shift_x]

            if shift_y < 0:
                size_range_y = [0 - shift_y, 200]
            else:
                size_range_y = [0, 200 - shift_y]

            if shift_z < 0:
                size_range_z = [0 - shift_z, 200]
            else:
                size_range_z = [0, 200 - shift_z]

            img_sroi = img_rot_copy[size_range_x[0]:size_range_x[1],
                                    size_range_y[0]:size_range_y[1],
                                    size_range_z[0]:size_range_z[1]]
            msk_sroi = msk_rot_copy[size_range_x[0]:size_range_x[1],
                                    size_range_y[0]:size_range_y[1],
                                    size_range_z[0]:size_range_z[1]]
            
            img_cube = img_sroi[31:159, 31:159, 31:159]
            msk_cube = msk_sroi[31:159, 31:159, 31:159]

            save_filename = "{}_bone{}_shift_{}_{}_{}_rotz_{}.nii.gz".format(
                ID, str(label), str(shift_x), str(shift_y), str(shift_z),
                str(rot_z))

            sitk.WriteImage(npimage_to_sitk(img_cube),
                            os.path.join(img_save_dir, save_filename))
            sitk.WriteImage(npimage_to_sitk(msk_cube),
                            os.path.join(msk_save_dir, save_filename))

def generate_label_classifier_datasets(
        ID, dataset_dir, save_dir):
    print(" ... processing ID: ", ID)
    check_path(save_dir)

    msk_savedir = os.path.join(save_dir)

    check_path(msk_savedir)

    _, pir_msk = read_image_and_mask(ID, dataset_dir)

    anno = read_annotation(ID, dataset_dir)
    locations = anno['locations']
    labels = anno['labels']

    pir_msk = (pir_msk > 0).astype(np.int)

    for i, (loc, label) in enumerate(zip(locations, labels)):
        
        msk_roi = get_ROI_img(pir_msk, loc)
        msk_roi = np.round(msk_roi).astype(np.int)

        assert len(np.unique(msk_roi)) == 2

        rot_range = [np.random.randint(-35, 35) for _ in range(2)]
        rot_range.append(0)

        for angle in rot_range:
            axes = [(1, 0), (2, 1), (2, 0)]
            axis = axes[np.random.randint(len(axes))]
            shift_x = np.random.randint(-20, 20)
            shift_y = np.random.randint(-20, 20)
            shift_z = np.random.randint(-20, 20)

            if angle == 0:
                shift_x, shift_y, shift_z = 0, 0, 0

            msk_rot = ndimage.rotate(msk_roi, angle, axis, cval=msk_roi.min(), reshape=False)
            msk_rot = np.round(msk_rot).astype(np.int)

            if shift_x < 0:
                size_range_x = [0 - shift_x, 200]
            else:
                size_range_x = [0, 200 - shift_x]

            if shift_y < 0:
                size_range_y = [0 - shift_y, 200]
            else:
                size_range_y = [0, 200 - shift_y]

            if shift_z < 0:
                size_range_z = [0 - shift_z, 200]
            else:
                size_range_z = [0, 200 - shift_z]


            msk_sroi = msk_rot[size_range_x[0]:size_range_x[1],
                               size_range_y[0]:size_range_y[1],
                               size_range_z[0]:size_range_z[1]]
            
            msk_cube = msk_sroi[31:159, 31:159, 31:159]

            save_filename = '{}_bone{}_shift_{}_{}_{}_rotz{}.nii.gz'.format(
                ID, str(label), str(shift_x), str(shift_y),
                str(shift_z), str(angle))

            sitk.WriteImage(npimage_to_sitk(msk_cube),
                            os.path.join(msk_savedir, save_filename))


def generate_spine_segmentor_datasets(ID, dataset_dir, save_dir):
    cube_size = (96, 96, 96)
    stride = 90
    stride_roi = 30

    check_path(save_dir)

    img_save_dir = os.path.join(save_dir, "img")
    msk_save_dir = os.path.join(save_dir, "msk")
    
    check_path(img_save_dir)
    check_path(msk_save_dir)


    print(" ... processing ID: ", ID)

    pir_img, pir_msk = read_image_and_mask(ID,
                                           dataset_dir)

    pir_img_copy = np.copy(pir_img)

    pir_msk[pir_msk > 0] = 1

    count = 0
    for i in range(2):
        axes = [(1, 0), (2, 1), (2, 0)]
        axis = axes[np.random.randint(len(axes))]
        angle = np.random.randint(-40, 40)
        if i == 0:
            angle = 0

        img_rot = ndimage.rotate(pir_img,
                                 angle,
                                 axis,
                                 cval=pir_img_copy.min(),
                                 reshape=False)

        msk_rot = ndimage.rotate(pir_msk, angle, axis, cval=0, reshape=False)
        msk_rot = np.round(msk_rot)

        img_cubes, msk_cubes = get_cubes(img_rot,
                                         msk_rot,
                                         cube_size,
                                         stride,
                                         norm=False)

        img_roi_cubes, msk_roi_cubes = get_roi_cubes(img_rot,
                                                     msk_rot,
                                                     cube_size,
                                                     stride_roi,
                                                     norm=False)

        img_cubes = np.concatenate((img_cubes, img_roi_cubes), axis=0)
        msk_cubes = np.concatenate((msk_cubes, msk_roi_cubes), axis=0)

        assert img_cubes.shape == msk_cubes.shape

        for idx, (img_cube, msk_cube) in enumerate(zip(img_cubes, msk_cubes)):
            assert img_cube.shape == msk_cube.shape

            if msk_cube.max() > 1:
                continue
            assert msk_cube.max() <= 1

            save_filename_img = os.path.join(
                img_save_dir, "{}_{:04d}.nii.gz".format(ID, count))
            save_filename_msk = os.path.join(
                msk_save_dir, "{}_{:04d}.nii.gz".format(ID, count))

            sitk.WriteImage(npimage_to_sitk(img_cube), save_filename_img)
            sitk.WriteImage(npimage_to_sitk(msk_cube), save_filename_msk)

            count += 1

def get_binary_msk_area(gt_msk):
    x_array, y_array, z_array = np.where(gt_msk > 0)
    min_x, max_x = min(x_array), max(x_array)
    min_y, max_y = min(y_array), max(y_array)
    min_z, max_z = min(z_array), max(z_array)
    area = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
    return area, (max_x - min_x, max_y - min_y, max_z - min_z)


def filter_msk_by_area(msk, thr=55000):
    labels = np.unique(msk)
    for label in labels:
        if label <= 0:
            continue
        msk_copy = np.copy(msk)
        gt_msk = (msk_copy == label).astype(int)
        area, _ = get_binary_msk_area(gt_msk)
        print(f"label = {label} area = {area}")
        if area < thr:
            msk[msk == label] = 0
    return msk


def filter_msk_by_count(msk, threshold=7820, threshold_ratio=0.5):
    labels = np.unique(msk)
    for label in labels:
        if label <= 0:
            continue
        msk_copy = np.copy(msk)
        gt_msk = (msk_copy == label).astype(int)
        count = np.count_nonzero(gt_msk)
        if count <= threshold * threshold_ratio:
            print(f"label {label} is low, delete...")
            msk[msk == label] = 0
    return msk


def proc_rot_msk(msk, angle, axis):
    labels = np.unique(msk)
    proc_msk = np.zeros(msk.shape)
    for label in labels:
        if label <= 0:
            continue
        msk_copy = np.copy(msk)
        gt_msk = (msk_copy == label).astype(np.int32)
        msk_rot = ndimage.rotate(gt_msk, angle, axis, cval=0, reshape=False)
        msk_rot = np.round(msk_rot)
        msk_rot[msk_rot == 1] = label
        proc_msk += msk_rot
        # proc_msk[msk_rot == 1] = label

    return proc_msk.astype(np.int32)


def make_binary_mask(mask):
    mask_copy = np.copy(mask)
    mask_copy[mask_copy != 0] = 1
    return mask_copy


def get_roi_box(binary_mask):
    binary_mask_copy = np.copy(binary_mask)
    binary_mask_copy = binary_erosion(binary_mask_copy).astype(
        binary_mask_copy.dtype)
    x_array, y_array, z_array = np.where(binary_mask_copy > 0)

    x_min, x_max = np.min(x_array), np.max(x_array)
    y_min, y_max = np.min(y_array), np.max(y_array)
    z_min, z_max = np.min(z_array), np.max(z_array)

    return x_min, x_max, y_min, y_max, z_min, z_max


def round_up(size, k=16):
    x = math.ceil(size / k)
    if x % 2 != 0:
        x = x + 1
    return x * k


def vol_padding_by_cube(img, cube_size, pad_value=-1024):
    img_pad = np.copy(img)

    cube_size_x, cube_size_y, cube_size_z = cube_size[:]

    pad_info = [(cube_size_x // 2, cube_size_x // 2),
                (cube_size_y // 2, cube_size_y // 2),
                (cube_size_z // 2, cube_size_z // 2)]

    img_pad = np.pad(img_pad,
                     pad_info,
                     mode='constant',
                     constant_values=pad_value).astype(np.float32)

    return img_pad, pad_info


def generate_idv_locate_datasets(ID, dataset_dir, save_folder, single_channel=True):
    
    check_path(save_folder)

    img_save_dir = os.path.join(save_folder, "img")
    mskheatmap_save_dir = os.path.join(save_folder, "mskheatmap")
    msk_save_dir = os.path.join(save_folder, "msk")

    check_path(img_save_dir)
    check_path(msk_save_dir)
    check_path(mskheatmap_save_dir)

    print(f"\n ... processing ID: {ID}\n")
    
    img, msk = read_image_and_mask(ID, dataset_dir, spacing=[2.0]*3)

    img, _ = vol_padding_by_cube(img, (72, 72, 72), img.min())
    msk, _ = vol_padding_by_cube(msk, (72, 72, 72), 0)

    binary_mask = make_binary_mask(msk)
    x_min, x_max, y_min, y_max, z_min, z_max = get_roi_box(binary_mask)

    x_edge, y_edge, z_edge = abs(x_min - x_max), abs(y_min -
                                                     y_max), abs(z_min - z_max)
    x_edge_up, y_edge_up, z_edge_up = round_up(x_edge), round_up(
        y_edge), round_up(z_edge)

    new_x_min, new_x_max = (x_max - x_min) // 2 + x_min - x_edge_up // 2, (
        x_max - x_min) // 2 + x_min + x_edge_up // 2
    new_y_min, new_y_max = (y_max - y_min) // 2 + y_min - y_edge_up // 2, (
        y_max - y_min) // 2 + y_min + y_edge_up // 2
    new_z_min, new_z_max = (z_max - z_min) // 2 + z_min - z_edge_up // 2, (
        z_max - z_min) // 2 + z_min + z_edge_up // 2

    img_crop = img[new_x_min:new_x_max, new_y_min:new_y_max,
                   new_z_min:new_z_max]
    msk_crop = msk[new_x_min:new_x_max, new_y_min:new_y_max,
                   new_z_min:new_z_max]

    angles = [np.random.randint(-30, 30) for _ in range(3)]
    angles.append(0)
    for angle in angles:
        axes = [(1, 0), (2, 1), (2, 0)]
        axis = axes[np.random.randint(len(axes))]

        img_rot = ndimage.rotate(img_crop,
                                 angle,
                                 axis,
                                 cval=img.min(),
                                 reshape=False)

        msk_rot = proc_rot_msk(msk_crop, angle, axis)
        msk_rot = filter_msk_by_count(msk_rot, 977)

        save_filename = f"{ID}_angel_{angle}_roted"

        sitk.WriteImage(
            npimage_to_sitk(img_rot),
            os.path.join(img_save_dir, save_filename + ".nii.gz"),
        )

        sitk.WriteImage(
            npimage_to_sitk(msk_rot),
            os.path.join(msk_save_dir, save_filename + ".nii.gz"),
        )

        heatmaps, heatmaps_sum = generator_heatmaps_by_msk(msk_rot,
                                             sigmas=3.0,
                                             num_landmarks=25)
        if single_channel:
            sitk.WriteImage(npimage_to_sitk(heatmaps_sum), 
                            os.path.join(mskheatmap_save_dir, save_filename + ".nii.gz"))
        else:
            np.savez_compressed(
                os.path.join(mskheatmap_save_dir, save_filename + ".npz"),
                heatmaps)


def generate_idv_fracture_classifier_datasets(ID, dataset_dir, save_dir):
    
    print(" ... processing ID: ", ID)

    check_path(save_dir)

    fracture_save_dir = os.path.join(save_dir, "fracture")
    normal_save_dir = os.path.join(save_dir, "normal")

    check_path(fracture_save_dir)
    check_path(normal_save_dir)

    vol_files = glob(os.path.join(dataset_dir, ID, "*.nii.gz"))

    for file in vol_files:
        if "idv_segment_mask" in file:
            msk_file = file
        elif "mask" not in file and "heatmap" not in file and "seg" not in file:
            img_file = file

    img = resample_to_spacing(sitk.ReadImage(img_file))
    msk = sitk.ReadImage(msk_file)

    pir_msk = sitk_to_npimage(msk)
    pir_img = sitk_to_npimage(img)

    anno = read_annotation(ID, dataset_dir)
    locations = anno["locations"]
    labels = anno["labels"]

    logs_file = glob(os.path.join(dataset_dir, ID, "log*.txt"))[0]
    idv_label = []
    with open(logs_file, encoding="gb18030") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "fracture" in line:
                label = line.split("=")[-1].split(",")
                for i in label:
                    idv_label.append(float(i))

    for index, (label, loc) in enumerate(zip(labels, locations)):
        gt_msk = (pir_msk == label).astype(np.int32)

        msk_roi = get_ROI_img(gt_msk, loc)
        msk_roi = np.round(msk_roi).astype(np.int32)

        img_roi = get_ROI_img(pir_img, loc)

        for i in range(4):
            axes = [(1, 0), (2, 1), (2, 0)]
            axis = axes[np.random.randint(len(axes))]
            angle = np.random.randint(-30, 30)
            shift_x = np.random.randint(-25, 25)
            shift_y = np.random.randint(-25, 25)
            shift_z = np.random.randint(-25, 25)

            if i == 1:
                angle, shift_x, shift_y, shift_z = 0, 0, 0, 0

            img_rot = ndimage.rotate(img_roi,
                                     angle,
                                     axis,
                                     cval=pir_img.min(),
                                     reshape=False)
            msk_rot = ndimage.rotate(msk_roi,
                                     angle,
                                     axis,
                                     cval=0,
                                     reshape=False)
            msk_rot = np.round(msk_rot).astype(np.int32)

            if shift_x < 0:
                size_range_x = [0 - shift_x, 200]
            else:
                size_range_x = [0, 200 - shift_x]

            if shift_y < 0:
                size_range_y = [0 - shift_y, 200]
            else:
                size_range_y = [0, 200 - shift_y]

            if shift_z < 0:
                size_range_z = [0 - shift_z, 200]
            else:
                size_range_z = [0, 200 - shift_z]

            img_sroi = img_rot[size_range_x[0]:size_range_x[1],
                               size_range_y[0]:size_range_y[1],
                               size_range_z[0]:size_range_z[1], ]

            msk_sroi = msk_rot[size_range_x[0]:size_range_x[1],
                               size_range_y[0]:size_range_y[1],
                               size_range_z[0]:size_range_z[1], ]

            x, y, z = center_of_mass(msk_sroi)
            crop_size = 96 // 2

            img_cube = img_sroi[int(x) - crop_size:int(x) + crop_size,
                                int(y) - crop_size:int(y) + crop_size,
                                int(z) - crop_size:int(z) + crop_size, ]

            msk_cube = msk_sroi[int(x) - crop_size:int(x) + crop_size,
                                int(y) - crop_size:int(y) + crop_size,
                                int(z) - crop_size:int(z) + crop_size, ]

            msk_cube = binary_erosion(msk_cube).astype(np.int32)

            single_cube = img_cube * msk_cube
            single_cube[single_cube == 0] = img_cube.min()

            image_stack = np.stack([single_cube, msk_cube],
                                   axis=0).astype(np.float32)

            if label in idv_label:
                save_filename = (
                    "fracture_{}_bone{}_shift_{}_{}_{}_rotz_{}.nii.gz".format(
                        ID,
                        str(label),
                        str(shift_x),
                        str(shift_y),
                        str(shift_z),
                        str(angle),
                    ))
                np.savez_compressed(
                    os.path.join(fracture_save_dir, save_filename),
                    image_stack)
            else:
                save_filename = "normal_{}_bone{}_shift_{}_{}_{}_rotz_{}.nii.gz".format(
                    ID, str(label), str(shift_x), str(shift_y), str(shift_z),
                    str(angle))
                np.savez_compressed(
                    os.path.join(normal_save_dir, save_filename), image_stack)


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_dataset(datadir):
    ID_list = os.listdir(datadir)
    d = {}
    for ID in ID_list:
        vol_files = glob(
            os.path.join(dataset_dir, ID, "{}*.nii.gz").format(ID))
        json_files = glob(os.path.join(dataset_dir, ID, "{}*.json").format(ID))
        if len(json_files) >= 2:
            for filename in json_files:
                filename_wo_folder = os.path.basename(filename)
                ext_length = len("_iso-ctd.json")
                filename_wo_folder_and_ext = filename_wo_folder[:-ext_length]
                d[filename_wo_folder_and_ext] = [
                    i for i in vol_files if filename_wo_folder_and_ext in i
                ] + [i for i in json_files if filename_wo_folder_and_ext in i]

    for item, values in d.items():
        dst_path = os.path.join(datadir, item)
        check_path(dst_path)
        for src_value in values:
            shutil.move(src_value, dst_path)

    for ID in ID_list:
        path = os.path.join(datadir, ID)
        if len(os.listdir(path)) == 0:
            os.rmdir(path)


def conut_labels(dataset_dir, task):
    train_ID_list = os.listdir(dataset_dir)
    vert_count = dict()
    for label in range(1, 31):
        vert_count[str(label)] = 0

    for ID in train_ID_list:
        anno = read_annotation(ID, dataset_dir)
        labels = anno["labels"]

        for label in labels:
            # label = 19 if label == 28 else label
            # label = 24 if label == 25 else label
            vert_count[str(label)] += 1

    save_path = os.path.join("data", task)
    check_path(save_path)
    save_filename = os.path.join(save_path,
                                 "classifier_num_of_each_label.json")
    with open(save_filename, 'w') as outfile:
        json.dump(vert_count, outfile, indent=4)  
    print('annotation saved to {}'.format(save_filename))



def creat_idv_dict():
    # 0-6: C1-C7
    # 7-18: T1-T12
    # 19-24: L1-L6
    # 25: T13
    idv_dict = {}
    for i in range(7):
        idv_dict[f"C{i+1}"] = i + 1
    for i in range(7, 19):
        idv_dict[f"T{i+1-7}"] = i + 1
    for i in range(19, 25):
        idv_dict[f"L{i+1-19}"] = i + 1
    idv_dict["T13"] = 26
    label_dict = dict(zip(idv_dict.values(), idv_dict.keys()))
    return idv_dict, label_dict


if __name__ == "__main__":
    task = "idv_segmentor"
    save_dir = task
    dataset_dir = "data/images_reoriented"

    check_dataset(dataset_dir)

    ID_list = os.listdir(dataset_dir)

    if task == "idv_segmentor":
        func = generate_idv_segmentor_datasets
        func_args = [(ID, dataset_dir, os.path.join("data", save_dir, ID)) for ID in ID_list]

    elif task == "classifier":
        conut_labels(dataset_dir, task)
        func = generate_label_classifier_datasets
        func_args = [(ID, dataset_dir, os.path.join("data", save_dir, ID)) for ID in ID_list]

    elif task == "spine_segmentor":
        func = generate_spine_segmentor_datasets
        func_args = [(ID, dataset_dir, os.path.join("data", save_dir, ID)) for ID in ID_list]

    elif task == "idv_locate_25_channel":
        func = generate_idv_locate_datasets
        func_args = [(ID, dataset_dir, os.path.join("data", save_dir, ID), False) for ID in ID_list]

    elif task == "idv_locate_1_channel":
        func = generate_idv_locate_datasets
        func_args = [(ID, dataset_dir, os.path.join("data", save_dir, ID), True) for ID in ID_list]

    elif task == "idv_fracture_classifier":
        func = generate_idv_fracture_classifier_datasets
        func_args = [(ID, dataset_dir, os.path.join("data", save_dir, ID)) for ID in ID_list]

    import multiprocessing
    pool = multiprocessing.Pool(6)
    pool.starmap(func, func_args)

    # for ID in ID_list:
    #     if len(os.listdir(os.path.join(dataset_dir, ID))) < 2:
    #         continue
    #     func(ID, dataset_dir, os.path.join("data", save_dir, ID))
