import SimpleITK as sitk
import numpy as np
import os
from scipy.ndimage import binary_erosion
import json
from utils.sitk_np import npimage_to_sitk, sitk_to_npimage


def convert_box_mode(box_info):

    x, y, z, w, h, d = box_info[:]

    xmin, xmax = x-w//2, x+w//2
    ymin, ymax = y-h//2, y+h//2
    zmin, zmax = z-d//2, z+d//2

    return xmin, ymin, zmin, xmax, ymax, zmax


def save_obj(vertices, faces, filename):
    with open(filename, "w") as f:
        for v in vertices:
            f.write("v {} {} {}\n".format(*np.array(v)))

        for t in faces:
            f.write("f {} {} {} {}\n".format(*(np.array(t) + 1)))


def set_vertices_faces(xmin, ymin, zmin, xmax, ymax, zmax, _i=0):
    vertices = []
    faces = []

    vertices += [
        (xmax, ymax, zmin),
        (xmax, ymin, zmin),
        (xmin, ymin, zmin),
        (xmin, ymax, zmin),
        (xmax, ymax, zmax),
        (xmax, ymin, zmax),
        (xmin, ymin, zmax),
        (xmin, ymax, zmax),
    ]

    faces += [
        (0 + 8 * _i, 1 + 8 * _i, 2 + 8 * _i, 3 + 8 * _i),
        (4 + 8 * _i, 7 + 8 * _i, 6 + 8 * _i, 5 + 8 * _i),
        (0 + 8 * _i, 4 + 8 * _i, 5 + 8 * _i, 1 + 8 * _i),
        (1 + 8 * _i, 5 + 8 * _i, 6 + 8 * _i, 2 + 8 * _i),
        (2 + 8 * _i, 6 + 8 * _i, 7 + 8 * _i, 3 + 8 * _i),
        (4 + 8 * _i, 0 + 8 * _i, 3 + 8 * _i, 7 + 8 * _i),
    ]

    return vertices, faces


def get_binary_area(binary_mask):
    binary_mask_copy = np.copy(binary_mask)
    binary_mask_copy = binary_erosion(
        binary_mask_copy).astype(binary_mask_copy.dtype)

    x_array, y_array, z_array = np.where(binary_mask_copy > 0)

    xmin, xmax = np.min(x_array), np.max(x_array)
    ymin, ymax = np.min(y_array), np.max(y_array)
    zmin, zmax = np.min(z_array), np.max(z_array)

    return xmin, ymin, zmin, xmax, ymax, zmax


def save_bbox(binary_mask, filename):
    xmin, ymin, zmin, xmax, ymax, zmax = get_binary_area(binary_mask)

    vertices, faces = set_vertices_faces(xmin, ymin, zmin, xmax, ymax, zmax)

    save_obj(vertices, faces, filename)


def save_bbox_center(binary_mask, filename):
    xmin, ymin, zmin, xmax, ymax, zmax = get_binary_area(binary_mask)

    box_size = [2.0] * 3

    x_center, y_center, z_center = (
        xmin + (xmax - xmin) / 2,
        ymin + (ymax - ymin) / 2,
        zmin + (zmax - zmin) / 2,
    )

    box_info = [x_center, y_center, z_center] + box_size

    vec = convert_box_mode(box_info)

    xmin, ymin, zmin = vec[0], vec[1], vec[2]
    xmax, ymax, zmax = vec[3], vec[4], vec[5]

    vertices, faces = set_vertices_faces(xmin, ymin, zmin, xmax, ymax, zmax)

    save_obj(vertices, faces, filename)


def save_landmark_point(landmark_file, save_folder):
    with open(landmark_file) as f:
        landmark = json.load(f)
    box_size = [2.0] * 3
    box_list = []
    labels = []

    for k in landmark:
        tmp = []
        tmp.append(k["X"])
        tmp.append(k["Y"])
        tmp.append(k["Z"])
        tmp += box_size
        box_list.append(tmp)
        labels.append(int(k["label"]))

    vertices = []
    faces = []

    for index, (label, _vec) in enumerate(zip(labels, box_list)):

        vec = convert_box_mode(_vec)
        xmin, ymin, zmin = vec[0], vec[1], vec[2]
        xmax, ymax, zmax = vec[3], vec[4], vec[5]

        _vertices, _faces = set_vertices_faces(
            xmin, ymin, zmin, xmax, ymax, zmax, index)
        vertices += _vertices
        faces += _faces
        
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    filename = os.path.join(save_folder, f"label_landmark.obj")
    save_obj(vertices=vertices, faces=faces, filename=filename)


if __name__ == "__main__":
    save_landmark_point(
        landmark_file="inference_result/GL003/landmark.json", save_folder="test_out"
    )
