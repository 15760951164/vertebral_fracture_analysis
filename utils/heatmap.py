import numpy as np
import scipy
import json
import SimpleITK as sitk
from scipy.ndimage import label, center_of_mass
from utils.sitk_np import sitk_to_npimage, npimage_to_sitk

def get_local_maxima(heatmap):

    dim = len(heatmap.shape)
    neigh = np.ones([3] * dim)
    neigh[tuple([1] * dim)] = 0
    filter = scipy.ndimage.maximum_filter(heatmap,
                                          footprint=neigh,
                                          mode='constant',
                                          cval=np.inf)

    maxima = (heatmap > filter)
    maxima_indizes = np.array(np.where(maxima))
    maxima_values = heatmap[tuple([maxima_indizes[i] for i in range(dim)])]

    return maxima_indizes.T, maxima_values


def generate_heatmap_target(heatmap_size,
                            landmarks,
                            sigmas,
                            scale=1.0,
                            normalize=False):

    landmarks_shape = landmarks.shape
    num_landmarks = landmarks_shape[1]
    dim = landmarks_shape[2] - 1

    landmarks_reshaped = np.reshape(
        landmarks[..., 1:], [num_landmarks] + [1] * dim + [dim])
    is_valid_reshaped = np.reshape(landmarks[..., 0],
                                   [num_landmarks] + [1] * dim)
    sigmas_reshaped = np.reshape(sigmas, [num_landmarks] + [1] * dim)

    aranges = [np.arange(heatmap_size[i]) for i in range(dim)]
    grid = np.meshgrid(*aranges, indexing='ij')

    grid_stacked = np.stack(grid, axis=dim).astype(np.float32)
    grid_stacked = np.stack([grid_stacked] * num_landmarks, axis=0)

    if normalize:
        scale /= np.power(np.sqrt(2 * np.pi) * sigmas_reshaped, dim)

    squared_distances = np.sum(np.power(grid_stacked - landmarks_reshaped,
                                        2.0),
                               axis=-1)
    heatmap = scale * np.exp(-squared_distances /
                             (2 * np.power(sigmas_reshaped, 2)))
    heatmap_or_zeros = np.where(
        (is_valid_reshaped + np.zeros_like(heatmap)) > 0, heatmap,
        np.zeros_like(heatmap))
    
    heatmap_or_zeros_sum = np.sum(heatmap_or_zeros, axis=0)

    return heatmap_or_zeros, heatmap_or_zeros_sum

def generator_one_heatmap(heatmap_size, coords, sigmas):
    coords = [[1, coords[0], coords[1], coords[2]]]
    landmark = np.expand_dims(np.array(coords), axis=0)
    heatmap = generate_heatmap_target(
        heatmap_size, landmark, [sigmas] * len(coords))[0][0]
    return heatmap

def get_msk_coords(msk):
    labels = np.unique(msk).astype(np.int8)
    coord_list = []
    for label in labels:
        if label <= 0:
            continue
        msk_copy = np.copy(msk)
        idv_mask = (msk_copy == label).astype(np.float32)
        x, y, z = center_of_mass(idv_mask)
        coord_list.append([1, x, y, z])
    return coord_list, labels

def generator_heatmaps_by_msk(msk, sigmas=3.0, num_landmarks=25):
    
    msk_roi_coords, labels = get_msk_coords(msk)
    landmark = np.expand_dims(np.array(msk_roi_coords), axis=0)
    msk_heatmaps, msk_heatmaps_sum = generate_heatmap_target(
        msk.shape, landmark, [sigmas] * len(msk_roi_coords))

    heatmaps = np.zeros(
        (num_landmarks, msk.shape[0], msk.shape[1], msk.shape[2]))

    for indices, label in enumerate(labels[1:]):
        if label < 1 or label > 25:
            continue
        heatmaps[label - 1] += msk_heatmaps[indices]
        
    return heatmaps, msk_heatmaps_sum


if __name__ == "__main__":

    coord_file = "test_data\GL195\GL195_CT_ax_iso-ctd.json"

    with open(coord_file, 'r') as f:
        data = f.read()
        coords = json.loads(data)
    coord_list = []

    for coord in coords:
        coord_list.append(np.array([1, coord["X"], coord["Y"], coord["Z"]]))
    
    landmark = np.expand_dims(np.array(coord_list), axis=0)

    image_path = "test_data\GL195\GL195_CT_ax.nii.gz"
    sitk_image = sitk.ReadImage(image_path)
    heatmap_size = np.ceil(
        np.array(sitk_image.GetSize()) *
        np.array(sitk_image.GetSpacing())).astype(int)
    

    heatmaps, heatmaps_sum = generate_heatmap_target(heatmap_size, landmark, [3.0] * len(coord_list))
    
    coord, value = get_local_maxima(heatmaps_sum)
    
    print(coord)
    
    sitk.WriteImage(npimage_to_sitk(heatmaps_sum), "test_out/heatmap_sum.nii.gz")