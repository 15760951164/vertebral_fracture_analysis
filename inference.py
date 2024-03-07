import torch
import os
import math
import time
import pickle
import copy
import timeout_decorator
import numpy as np
import SimpleITK as sitk
from skimage import filters
import networkx as nx
import matplotlib.pyplot as plt
from utils.sitk_np import npimage_to_sitk, sitk_to_npimage
from utils.sitk_image import resample_to_spacing
from utils.heatmap import get_local_maxima, generator_one_heatmap
from models.ResNet import generate_resnet_model
from scipy.ndimage import center_of_mass, binary_erosion, label
from models.UNet import *
from models import SCN_Modify


class Landmark(object):

    def __init__(self,
                 coords=None,
                 is_valid=True,
                 scale=1.0,
                 value=0,
                 label=-1):

        self.coords = coords
        self.is_valid = is_valid
        if self.is_valid is None:
            self.is_valid = self.coords is not None
        self.scale = scale
        self.value = value
        self.label = label

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other) -> bool:
        return self.label == other.label


class LandmarkGraphOptimization(object):

    def __init__(self, num_landmarks, bias=3.0, l=0.3):

        self.num_landmarks = num_landmarks
        self.bias = bias
        self.l = l

    def unary_term(self, landmark):

        return self.l * landmark.value + self.bias

    def pairwise_term(self, landmark_a, landmark_b):

        distance_value = np.linalg.norm(landmark_a.coords - landmark_b.coords)

        return (1 - self.l) * distance_value

    def remove_valid_landmark(self, local_heatmap_maxima):
        new_local_heatmap_maxima = []
        for landmark_list in local_heatmap_maxima:
            if len(landmark_list) > 0 and landmark_list[0].is_valid == True:
                new_local_heatmap_maxima.append(landmark_list)
        return new_local_heatmap_maxima

    def create_graph(self, local_heatmap_maxima):

        G = nx.DiGraph()

        new_local_heatmap_maxima = self.remove_valid_landmark(
            local_heatmap_maxima)

        for curr in range(len(new_local_heatmap_maxima) - 1):
            next = curr + 1
            curr_landmarks = new_local_heatmap_maxima[curr]
            next_landmarks = new_local_heatmap_maxima[next]
            for curr_index, curr_landmark in enumerate(curr_landmarks):
                for next_index, next_landmark in enumerate(next_landmarks):

                    weight = self.pairwise_term(
                        curr_landmark,
                        next_landmark) - self.unary_term(curr_landmark)

                    G.add_edge(f'{curr_landmark.label-1}_{curr_index}',
                               f'{next_landmark.label-1}_{next_index}',
                               weight=weight)

        for cur_index, curr_landmark in enumerate(new_local_heatmap_maxima[0]):
            G.add_edge('s',
                       f'{curr_landmark.label-1}_{cur_index}',
                       weight=self.unary_term(curr_landmark))

        for cur_index, curr_landmark in enumerate(
                new_local_heatmap_maxima[-1]):
            G.add_edge(f'{curr_landmark.label-1}_{cur_index}',
                       't',
                       weight=self.unary_term(curr_landmark))

        return G

    def vertex_name_to_indizes(self, name):

        landmark_index = int(name[:name.find('_')])
        maxima_index = int(name[name.find('_') + 1:])
        return landmark_index, maxima_index

    def path_to_landmarks(self, path, local_heatmap_maxima):

        landmarks = [
            Landmark(coords=[np.nan] * 3, is_valid=False)
            for _ in range(self.num_landmarks)
        ]
        for node in path:
            if node == 's' or node == 't':
                continue
            landmark_index, maxima_index = self.vertex_name_to_indizes(node)
            landmarks[landmark_index] = local_heatmap_maxima[landmark_index][
                maxima_index]
        return landmarks

    def graph_optimization(self, local_heatmap_maxima, graph_filename=None):

        G = self.create_graph(local_heatmap_maxima)
        if graph_filename is not None:
            import matplotlib.pyplot as plt
            nx.draw(G,
                    with_labels=True,
                    node_color='skyblue',
                    node_size=1000,
                    font_size=10,
                    font_weight='bold')
            plt.savefig(graph_filename, dpi=300)
        shortest_path = nx.shortest_path(G,
                                         's',
                                         't',
                                         'weight',
                                         method='bellman-ford')
        distances = []
        for i in range(1, len(shortest_path) - 2):
            edge = G.edges[shortest_path[i], shortest_path[i + 1]]
            weight = edge['weight']
            distances.append(
                (f'{shortest_path[i]}_{shortest_path[i+1]}', f'{weight:0.4f}'))
        return self.path_to_landmarks(shortest_path, local_heatmap_maxima)


class Box(object):

    def __init__(self, min_coords, max_coords) -> None:
        self.min_coords = min_coords
        self.max_coords = max_coords

    def get_area(self):
        min_x, min_y, min_z = self.min_coords[:]
        max_x, max_y, max_z = self.max_coords[:]

        return (max_x - min_x) * (max_y - min_y) * (max_z - min_z)

    def get_x_range(self):
        min_x, _, _ = self.min_coords[:]
        max_x, _, _ = self.max_coords[:]

        return max_x - min_x

    def get_y_range(self):
        _, min_y, _ = self.min_coords[:]
        _, max_y, _ = self.max_coords[:]

        return max_y - min_y

    def get_z_range(self):
        _, _, min_z = self.min_coords[:]
        _, _, max_z = self.max_coords[:]

        return max_z - min_z

    def get_center_point(self):

        min_x, min_y, min_z = self.min_coords[:]

        return np.array([
            self.get_x_range() // 2 + min_x,
            self.get_y_range() // 2 + min_y,
            self.get_z_range() // 2 + min_z
        ])

    def get_edge_range(self):

        min_x, min_y, min_z = self.min_coords[:]
        max_x, max_y, max_z = self.max_coords[:]

        return abs(max_x - min_x), abs(max_y - min_y), abs(max_z - min_z)


class InferenceBase(object):

    def __init__(self, model_file=None, model_func=None) -> None:
        self.model_func = model_func
        self.model_file = model_file
        self.cube_size = (96, 96, 96)
        self.stride = 24
        self.input_image_spacing = None
        self.model_dict = {}
        self.display_iter = 0

        self.load_model()

    def load_model(self):
        if self.model_file is None or self.model_func is None:
            return
        state_dict = torch.load(self.model_file)["state_dict"]
        # state_dict = torch.load(self.model_file)
        self.model_func.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self.model_func.to(torch.device("cuda"))
        self.model_func.eval()
        print(f"Model {self.model_func.__class__.__name__} Load Key Finish...")

    def set_cube_size(self, new_cube_size):
        self.cube_size = new_cube_size

    def read_image(self, image_path, out_spacing=[1.0, 1.0, 1.0]):
        if image_path is None:
            return
        sitk_image = sitk.ReadImage(image_path)
        self.input_image_spacing = sitk_image.GetSpacing()

        resampled_image = resample_to_spacing(sitk_image, out_spacing)
        return sitk_to_npimage(resampled_image)

    def crop_by_coords(self, image, coords):
        crop_image_list = []

        image_pad, pad_info = self.vol_padding_by_cube(image)

        for coord_info in coords:
            if coord_info.is_valid == False:
                continue
            coord = coord_info.coords.astype(np.int32)

            x = coord[0] + pad_info[0][0]
            y = coord[1] + pad_info[1][0]
            z = coord[2] + pad_info[2][0]

            x_start, x_end = x - self.cube_size[0] // 2, x + self.cube_size[
                0] // 2
            y_start, y_end = y - self.cube_size[1] // 2, y + self.cube_size[
                1] // 2
            z_start, z_end = z - self.cube_size[2] // 2, z + self.cube_size[
                2] // 2

            crop_image = image_pad[x_start:x_end, y_start:y_end, z_start:z_end]

            assert (np.array(crop_image.shape) == np.array(
                self.cube_size)).all()

            crop_image_list.append(crop_image)

        return crop_image_list

    def vol_padding_by_cube(self, img):
        img_pad = np.copy(img)

        cube_size_x, cube_size_y, cube_size_z = self.cube_size[:]

        pad_info = [(cube_size_x // 2, cube_size_x // 2),
                    (cube_size_y // 2, cube_size_y // 2),
                    (cube_size_z // 2, cube_size_z // 2)]

        img_pad = np.pad(img_pad,
                         pad_info,
                         mode='constant',
                         constant_values=img.min()).astype(np.float32)

        return img_pad, pad_info

    def vol_padding(self, vol, pad_value=-1024, front_after_pad_size=24):

        h, w, c = vol.shape
        padX = [front_after_pad_size, front_after_pad_size]
        padY = [front_after_pad_size, front_after_pad_size]
        padZ = [front_after_pad_size, front_after_pad_size]

        if h < self.cube_size[0]:
            padX[1] += self.cube_size[0] - h
        elif h % self.cube_size[0] != 0:
            padX[1] += self.cube_size[0] - (h % self.cube_size[0])

        if w < self.cube_size[1]:
            padY[1] += self.cube_size[1] - w
        elif w % self.cube_size[1] != 0:
            padY[1] += self.cube_size[1] - (w % self.cube_size[1])

        if c < self.cube_size[2]:
            padZ[1] += self.cube_size[2] - c
        elif c % self.cube_size[2] != 0:
            padZ[1] += self.cube_size[2] - (c % self.cube_size[2])

        vol = np.pad(vol, (tuple(padX), tuple(padY), tuple(padZ)),
                     mode='constant',
                     constant_values=pad_value).astype(np.float32)

        return vol, [padX, padY, padZ]

    def to_tensor(self, x, use_cuda=True):
        x = torch.from_numpy(x)
        x = x.to(dtype=torch.float32)
        if use_cuda and torch.cuda.is_available():
            x = x.cuda()
        return x

    def to_numpy(self, x):
        x = x.cpu()
        x = x.detach().numpy().astype(np.float32)
        return x

    def save_npimage(self,
                     image,
                     save_path,
                     resmaple=False,
                     out_spacing=[1.0] * 3):
        print(f"save inference result to {save_path}...")
        sitk_image = npimage_to_sitk(image)
        if resmaple and out_spacing is not None:
            sitk_image = resample_to_spacing(sitk_image, out_spacing)
        sitk.WriteImage(sitk_image, save_path)

    def display_info(self, info, step):
        if self.display_iter % step == 0:
            print(info)
        self.display_iter += 1

    def get_roi_box(self, binary_mask):

        binary_mask_copy = np.copy(binary_mask)
        binary_mask_copy = binary_erosion(binary_mask_copy).astype(
            binary_mask_copy.dtype)
        x_array, y_array, z_array = np.where(binary_mask_copy > 0)

        x_min, x_max = np.min(x_array), np.max(x_array)
        y_min, y_max = np.min(y_array), np.max(y_array)
        z_min, z_max = np.min(z_array), np.max(z_array)

        roi_box = Box(min_coords=np.array([x_min, y_min, z_min]),
                      max_coords=np.array([x_max, y_max, z_max]))

        return roi_box

    def overlapping_patch_inference(self, image):

        ori_h, ori_w, ori_c = image.shape
        image_pad, pad_info = self.vol_padding(image)
        h, w, c = image_pad.shape

        vol_out = np.zeros(image_pad.shape, dtype=np.float32)
        idx_vol = np.zeros(image_pad.shape, dtype=np.float32)

        h_scaning = ((h - self.cube_size[0]) // self.stride) + 1
        w_scaning = ((w - self.cube_size[1]) // self.stride) + 1
        c_scaning = ((c - self.cube_size[2]) // self.stride) + 1
        total = int(h_scaning * w_scaning * c_scaning)

        for i in range(0, h, self.stride):
            for j in range(0, w, self.stride):
                for k in range(0, c, self.stride):

                    x_start, x_end = i, i + self.cube_size[0]
                    y_start, y_end = j, j + self.cube_size[1]
                    z_start, z_end = k, k + self.cube_size[2]

                    if x_end > h or y_end > w or z_end > c:
                        continue

                    self.display_info(
                        info=
                        f"overlapping path scanning at {self.display_iter}/{total} --> {i, j, k}...",
                        step=50)

                    cube = image_pad[x_start:x_end, y_start:y_end,
                                     z_start:z_end]

                    cube = self.to_tensor(cube)
                    cube = cube.view(1, 1, cube.shape[0], cube.shape[1],
                                     cube.shape[2])
                    pred = self.model_func(cube)
                    cube_out = np.squeeze(self.to_numpy(pred))

                    vol_out[x_start:x_end, y_start:y_end,
                            z_start:z_end] += cube_out
                    idx_vol[x_start:x_end, y_start:y_end, z_start:z_end] += 1

        vol_out = vol_out / idx_vol
        vol_out = np.nan_to_num(vol_out)

        vol_output = vol_out[pad_info[0][0]:pad_info[0][0] + ori_h,
                             pad_info[1][0]:pad_info[1][0] + ori_w,
                             pad_info[2][0]:pad_info[2][0] + ori_c]

        return vol_output

    @torch.no_grad()
    def inference(self, inference_image=None, save_path=None):
        pass


class SpineSegmention(InferenceBase):

    def __init__(
        self,
        model_file,
        model_func=UNet3D_ResidualSE(in_channels=1,
                                     out_channels=1,
                                     f_maps=32,
                                     layer_order="cbr",
                                     repeats=1,
                                     final_activation="sigmoid",
                                     conv_kernel_size=3,
                                     conv_padding=1,
                                     use_attn=False,
                                     num_levels=5)
    ) -> None:
        super().__init__(model_file, model_func)

        self.cube_size = (96, 96, 96)
        self.front_after_pad_size = 24
        self.stride = 32

    def get_binary_mask(self, image=None):

        vol_input = np.copy(image)

        assert vol_input is not None

        vol_output = self.overlapping_patch_inference(vol_input)

        thr = filters.threshold_otsu(vol_output)

        vol_output[vol_output > thr] = 1
        vol_output[vol_output <= thr] = 0

        return vol_output

    @torch.no_grad()
    def inference(self, inference_image=None, save_path=None):
        torch.cuda.empty_cache()
        self.display_iter = 1
        if isinstance(inference_image, str):
            vol_input = self.read_image(inference_image)
        else:
            vol_input = inference_image

        strat_time = time.time()
        print("\nbinary mask inference start...")

        binary_mask = self.get_binary_mask(vol_input)

        roi_box = self.get_roi_box(binary_mask)

        if save_path is not None:
            self.save_npimage(binary_mask,
                              os.path.join(save_path, "binary_mask.nii.gz"),
                              resmaple=True)

        end_time = time.time()
        print(
            f"binary mask inference done..., elapsed time {(end_time-strat_time):.2f} secs"
        )

        return binary_mask, roi_box


class VertebraLocate_Base(InferenceBase):

    def __init__(self,
                 num_labels=None,
                 model_file=None,
                 model_func=None) -> None:
        super().__init__(model_file, model_func)

        self.num_labels = num_labels

    def inference_roi_image(self, image: np.ndarray, roi_box_ori: Box,
                            roi_box_up: Box):

        vol_output = np.zeros(
            (self.num_labels, image.shape[0], image.shape[1], image.shape[2]))

        image_pad, pad_info = self.vol_padding_by_cube(image)

        x_min_up, y_min_up, z_min_up = roi_box_up.min_coords[:]
        x_max_up, y_max_up, z_max_up = roi_box_up.max_coords[:]

        new_xmin, new_ymin, new_zmin = x_min_up + pad_info[0][
            0], y_min_up + pad_info[1][0], z_min_up + pad_info[2][0]
        new_xmax, new_ymax, new_zmax = x_max_up + pad_info[0][
            0], y_max_up + pad_info[1][0], z_max_up + pad_info[2][0]

        cube = image_pad[new_xmin:new_xmax, new_ymin:new_ymax,
                         new_zmin:new_zmax]

        cube = self.to_tensor(cube)
        cube = cube.view(1, 1, cube.shape[0], cube.shape[1], cube.shape[2])
        pred = self.model_func(cube)
        cube_out = np.squeeze(self.to_numpy(pred))

        x_min_ori, y_min_ori, z_min_ori = roi_box_ori.min_coords[:]
        x_max_ori, y_max_ori, z_max_ori = roi_box_ori.max_coords[:]

        x_min_crop = abs(x_min_up - x_min_ori)
        x_max_crop = x_min_crop + roi_box_ori.get_x_range()

        y_min_crop = abs(y_min_up - y_min_ori)
        y_max_crop = y_min_crop + roi_box_ori.get_y_range()

        z_min_crop = abs(z_min_up - z_min_ori)
        z_max_crop = z_min_crop + roi_box_ori.get_z_range()

        vol_output[..., x_min_ori:x_max_ori, y_min_ori:y_max_ori, z_min_ori:z_max_ori] += \
            cube_out[..., x_min_crop:x_max_crop, y_min_crop:y_max_crop, z_min_crop:z_max_crop]

        vol_output = np.nan_to_num(vol_output)

        return vol_output, np.sum(vol_output, axis=0)

    def round_up(self, size, k=16):
        x = math.ceil(size / k)
        if x % 2 != 0:
            x = x + 1
        return x * k

    def box_round_up(self, roi_box: Box):

        x_min, y_min, z_min = roi_box.min_coords[:]
        x_max, y_max, z_max = roi_box.max_coords[:]

        x_edge, y_edge, z_edge = roi_box.get_edge_range()

        x_edge_up, y_edge_up, z_edge_up = self.round_up(x_edge), self.round_up(
            y_edge), self.round_up(z_edge)

        new_x_min, new_x_max = (x_max - x_min) // 2 + x_min - x_edge_up // 2, (
            x_max - x_min) // 2 + x_min + x_edge_up // 2
        new_y_min, new_y_max = (y_max - y_min) // 2 + y_min - y_edge_up // 2, (
            y_max - y_min) // 2 + y_min + y_edge_up // 2
        new_z_min, new_z_max = (z_max - z_min) // 2 + z_min - z_edge_up // 2, (
            z_max - z_min) // 2 + z_min + z_edge_up // 2

        return Box(min_coords=np.array([new_x_min, new_y_min, new_z_min]),
                   max_coords=np.array([new_x_max, new_y_max, new_z_max]))

    def convert_roi_box_to_2mm(self, roi_box):
        return Box(roi_box.min_coords // 2, roi_box.max_coords // 2)

    def convert_coords_to_1mm(self, coords_info):
        new_coords = copy.deepcopy(coords_info)
        for i in range(len(new_coords)):
            new_coords[i].coords *= 2
        return new_coords

    def inference_heatmap(self, inference_image=None, roi_box: Box = None):

        vol_input = inference_image

        assert vol_input is not None

        roi_box_ori = self.convert_roi_box_to_2mm(roi_box)
        roi_box_up = self.box_round_up(roi_box_ori)

        vol_output, heatmaps = self.inference_roi_image(
            vol_input, roi_box_ori, roi_box_up)

        return vol_output, heatmaps


class VertebraLocate_25_Channel(VertebraLocate_Base):

    def __init__(
        self,
        min_landmark_value=0.3,
        num_labels=25,
        postprocess_func=LandmarkGraphOptimization,
        model_file=None,
        model_func=SCN_Modify.SCN(in_channels=1,
                                  out_channels=25,
                                  f_maps=64,
                                  dropout_prob=0.3)
    ) -> None:
        super().__init__(num_labels, model_file, model_func)

        self.postprocess = postprocess_func
        self.num_labels = num_labels
        self.min_landmark_value = min_landmark_value

    def inference_aux(self, image, roi_box):

        vol_output, heatmaps = self.inference_heatmap(image, roi_box)

        coords_list = []
        for i in range(vol_output.shape[0]):
            if np.max(vol_output[i]) < self.min_landmark_value:
                coords_list.append([
                    Landmark(coords=np.array([np.nan] * 3),
                             is_valid=False,
                             label=i + 1,
                             scale=1,
                             value=0)
                ])
                continue

            vol_copy = np.copy(vol_output[i])
            th = filters.threshold_otsu(vol_copy)
            vol_copy = np.clip(vol_copy, th, np.max(vol_copy))
            coords, values = get_local_maxima(vol_copy)

            coords_list.append([
                Landmark(coords=coord,
                         is_valid=True,
                         scale=1,
                         label=i + 1,
                         value=value) for coord, value in zip(coords, values)
            ])

        return heatmaps, coords_list

    def filter_landmarks_top_bottom(self,
                                    coords_info,
                                    image_size,
                                    z_distance_top_bottom=10):
        filtered_landmarks = []
        for l in coords_info:
            if z_distance_top_bottom < l.coords[
                    2] < image_size[2] - z_distance_top_bottom:
                filtered_landmarks.append(l)
            else:
                filtered_landmarks.append(
                    Landmark(coords=[np.nan] * 3, is_valid=False))
        return filtered_landmarks

    def add_mean_coords(self, coords_info):
        for coord_list in coords_info:
            if len(coord_list) != 1:
                m_coord_list = []
                m_value_list = []
                for coord in coord_list:
                    if coord.is_valid == True:
                        m_coord_list.append(coord.coords)
                        m_value_list.append(coord.value)
                coord_list.append(
                    Landmark(coords=np.stack(
                        m_coord_list, axis=0).mean(axis=0).astype(np.int32),
                             is_valid=True,
                             scale=1.0,
                             label=coord_list[0].label,
                             value=np.stack(m_value_list,
                                            axis=0).mean(axis=0)))
        return coords_info

    def remove_valid_landmark(self, coords_info):
        new_coord = []
        for coord in coords_info:
            if coord.is_valid == False:
                continue
            new_coord.append(coord)
        return new_coord

    #@timeout_decorator.timeout(30)
    def graph_optimization(self, coords_info, vol_input):

        print("using graph optimization on coords...")

        postprocess = self.postprocess(num_landmarks=self.num_labels,
                                       bias=2.0,
                                       l=0.2)

        coords_info = self.add_mean_coords(coords_info)
        coords_info = postprocess.graph_optimization(coords_info,
                                                     graph_filename=None)

        coords_info = self.filter_landmarks_top_bottom(coords_info,
                                                       vol_input.shape)
        coords_info = self.remove_valid_landmark(coords_info)

        return coords_info

    def filter_coords_by_roi(self, coords_info, roi_box: Box):
        new_coords = []
        for coord in coords_info:
            if np.all(coord.coords > roi_box.min_coords) and np.all(
                    coord.coords < roi_box.max_coords):
                new_coords.append(coord)
        return new_coords

    @torch.no_grad()
    def inference(self,
                  inference_image=None,
                  save_path=None,
                  roi_box: Box = None):
        torch.cuda.empty_cache()

        if isinstance(inference_image, str):
            vol_input = self.read_image(inference_image, out_spacing=[2.0] * 3)
        else:
            vol_input = inference_image

        assert vol_input is not None

        strat_time = time.time()
        print("\nidv_locate inference start...")

        heatmap, coords_info = self.inference_aux(vol_input, roi_box)

        if self.postprocess is not None:
            try:
                coords_info = self.graph_optimization(coords_info, vol_input)
            except timeout_decorator.TimeoutError:
                print("graph optimization falid, continue...")
                coords_info = [coords[-1] for coords in coords_info]

        coords_info = self.convert_coords_to_1mm(coords_info)

        if save_path is not None:
            self.save_npimage(heatmap,
                              os.path.join(save_path,
                                           "heatmap_25_channel.nii.gz"),
                              resmaple=True)

        end_time = time.time()
        print(
            f"idv_locate inference done..., elapsed time {(end_time-strat_time):.2f} secs"
        )

        return heatmap, coords_info


class VertebraLocate_1_Channel(VertebraLocate_Base):

    def __init__(
        self,
        min_landmark_value=0.3,
        num_labels=1,
        model_file=None,
        model_func=SCN_Modify.SCN(in_channels=1,
                                  out_channels=1,
                                  f_maps=32,
                                  dropout_prob=0.2)
    ) -> None:
        super().__init__(num_labels, model_file, model_func)

        self.num_labels = num_labels
        self.min_landmark_value = min_landmark_value

    def inference_aux(self, image, roi_box):

        vol_output, heatmaps = self.inference_heatmap(image, roi_box)

        vol_output = np.squeeze(vol_output)

        coords_list = []
        vol_copy = np.copy(vol_output)
        th = filters.threshold_otsu(vol_copy)
        vol_copy = np.clip(vol_copy, th, np.max(vol_copy))
        coords, values = get_local_maxima(vol_copy)

        for coord, value in zip(coords, values):
            if value < self.min_landmark_value:
                continue
            coords_list.append(
                Landmark(coords=coord,
                         is_valid=True,
                         scale=1,
                         label=-1,
                         value=value))

        coords_list = sorted(coords_list,
                             key=lambda x: x.coords[2],
                             reverse=True)

        return heatmaps, coords_list

    @torch.no_grad()
    def inference(self,
                  inference_image=None,
                  save_path=None,
                  roi_box: Box = None):

        torch.cuda.empty_cache()

        if isinstance(inference_image, str):
            vol_input = self.read_image(inference_image, out_spacing=[2.0] * 3)
        else:
            vol_input = inference_image

        assert vol_input is not None

        strat_time = time.time()
        print("\nidv_locate inference start...")

        heatmap, coords_info = self.inference_aux(vol_input, roi_box)

        coords_info = self.convert_coords_to_1mm(coords_info)

        if save_path is not None:
            self.save_npimage(heatmap,
                              os.path.join(save_path,
                                           "heatmap_1_channel.nii.gz"),
                              resmaple=True)

        end_time = time.time()
        print(
            f"idv_locate inference done..., elapsed time {(end_time-strat_time):.2f} secs"
        )

        return heatmap, coords_info


class VertebraSegmention(InferenceBase):

    def __init__(
        self,
        model_file=None,
        model_func=UNet3D_ResidualSE(in_channels=2,
                                    out_channels=1,
                                    f_maps=32,
                                    layer_order="cbrd",
                                    dropout_prob=0.25,
                                    repeats=1,
                                    final_activation="sigmoid",
                                    conv_kernel_size=3,
                                    conv_padding=1,
                                    use_attn=True,
                                    num_levels=5)
    ) -> None:
        super().__init__(model_file, model_func)

        self.cube_size = (128, 128, 128)

    def filter_msk_by_area(self, msk, thr=55000):
        labels = np.unique(msk)
        for label in labels:
            if label <= 0:
                continue
            msk_copy = np.copy(msk)
            gt_msk = (msk_copy == label).astype(int)
            x_array, y_array, z_array = np.where(gt_msk > 0)
            min_x, max_x = min(x_array), max(x_array)
            min_y, max_y = min(y_array), max(y_array)
            min_z, max_z = min(z_array), max(z_array)
            area = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
            print(f"label = {label} area = {area}")
            if area < thr:
                msk[msk == label] = 0
        return msk

    def filter_msk_by_count(self, msk, threshold=7820, threshold_ratio=0.6):
        filter_count = 0
        labels = np.unique(msk)

        for label in labels:
            if label <= 0:
                continue
            msk_copy = np.copy(msk)
            gt_msk = (msk_copy == label).astype(int)
            count = np.count_nonzero(gt_msk)
            # print(f"label = {label} area = {count}")
            if count <= threshold * threshold_ratio:
                print(f"label {label} segment is too low, remove...")
                msk[msk == label] = 0
                filter_count += 1

        return msk, filter_count

    def make_labels_continuous(self, muti_label_mask):
        muti_label_mask_copy = np.copy(muti_label_mask)
        labels = np.unique(muti_label_mask)
        prev = None
        prev_count = 0
        for label in labels:
            if label == 0:
                continue
            if prev is not None:
                l = abs(label - prev)
                if l != 1:
                    prev_count += l - 1
            muti_label_mask_copy[muti_label_mask_copy ==
                                 label] = label - prev_count
            prev = label
        return muti_label_mask_copy

    def combine_mask_list(self, vertebra_mask_list, is_filter):

        multi_label_mask = np.zeros(vertebra_mask_list[0]["mask"].shape,
                                    dtype=np.float32)

        for index, mask_info in enumerate(vertebra_mask_list):
            mask = mask_info["mask"]
            label = mask_info["label"]
            if label == -1:
                label = 50 + index
            multi_label_mask[mask == 1] = label

        if is_filter:
            multi_label_mask, filter_count = self.filter_msk_by_count(
                multi_label_mask)
            # for _ in range(filter_count):
            #     multi_label_mask = self.make_labels_continuous(
            #         multi_label_mask)

        return multi_label_mask

    @torch.no_grad()
    def inference(self,
                  inference_image=None,
                  save_path=None,
                  coord_info=None,
                  filter=False):

        torch.cuda.empty_cache()
        self.display_iter = 1
        if isinstance(inference_image, str):
            vol_input = self.read_image(inference_image)
        else:
            vol_input = inference_image

        assert vol_input is not None or coord_info is not None

        strat_time = time.time()
        print("\nidv_segment inference start...")

        ori_h, ori_w, ori_c = vol_input.shape
        image_pad, pad_info = self.vol_padding_by_cube(vol_input)
        muti_label_mask = np.zeros(image_pad.shape, dtype=np.float32)

        h, w, c = image_pad.shape
        vertebra_mask_list = []
        for index, landmark in enumerate(coord_info):
            if landmark.is_valid == False:
                continue
            vol_output = np.zeros(image_pad.shape, dtype=np.float32)
            coords = landmark.coords.astype(np.int32)
            x = coords[0] + pad_info[0][0]
            y = coords[1] + pad_info[1][0]
            z = coords[2] + pad_info[2][0]

            x_start, x_end = x - self.cube_size[0] // 2, x + self.cube_size[
                0] // 2
            y_start, y_end = y - self.cube_size[1] // 2, y + self.cube_size[
                1] // 2
            z_start, z_end = z - self.cube_size[2] // 2, z + self.cube_size[
                2] // 2

            if x_end > h or y_end > w or z_end > c or z_start < 0:
                continue

            self.display_info(
                f"idv_segment at coords {index + 1}/{len(coord_info)}_{landmark.label} --> {coords}...",
                step=1)

            crop_image = image_pad[x_start:x_end, y_start:y_end, z_start:z_end]

            assert (np.array(crop_image.shape) == np.array(
                self.cube_size)).all()

            center_point = (self.cube_size[0] // 2, self.cube_size[1] // 2,
                            self.cube_size[2] // 2)
            heatmap = generator_one_heatmap(heatmap_size=self.cube_size,
                                            coords=center_point,
                                            sigmas=3.0)

            cube = np.stack([crop_image, heatmap], axis=0)

            pred = self.model_func(self.to_tensor(cube).unsqueeze(dim=0))
            cube_out = np.squeeze(self.to_numpy(pred))

            #thr = filters.threshold_otsu(cube_out)
            thr = 0.5
            cube_out[cube_out > thr] = 1
            cube_out[cube_out <= thr] = 0

            vol_output[x_start:x_end, y_start:y_end, z_start:z_end] = cube_out

            idv_mask = vol_output[pad_info[0][0]:pad_info[0][0] + ori_h,
                                  pad_info[1][0]:pad_info[1][0] + ori_w,
                                  pad_info[2][0]:pad_info[2][0] + ori_c]

            vertebra_mask_list.append({
                "label": landmark.label,
                "mask": idv_mask,
                "image_input": cube,
                "image_out": cube_out,
                "center_coords": np.array(list(center_of_mass(idv_mask)))
            })

        muti_label_mask = self.combine_mask_list(vertebra_mask_list, filter)

        end_time = time.time()
        print(
            f"idv_segment inference done..., elapsed time {(end_time-strat_time):.2f} secs"
        )

        if save_path is not None:
            self.save_npimage(muti_label_mask,
                              os.path.join(save_path,
                                           "idv_segment_mask.nii.gz"),
                              resmaple=True)

        return muti_label_mask, vertebra_mask_list


class VertebraFractureClassifier(InferenceBase):

    def __init__(
        self,
        model_file,
        model_func=generate_resnet_model(n_input_channels=2,
                                         model_depth=50,
                                         n_classes=2)
    ) -> None:
        super().__init__(model_file, model_func)

        self.cube_size = (96, 96, 96)

    def inference_image_type(self, input, stack):

        if stack:
            input = self.to_tensor(input).view(1, input.shape[0],
                                               input.shape[1], input.shape[2],
                                               input.shape[3])
        else:
            input = self.to_tensor(input).view(1, 1, input.shape[0],
                                               input.shape[1], input.shape[2])

        pred = self.model_func(input)

        class_type = torch.argmax(pred)
        class_prob = torch.nn.functional.softmax(pred, dim=1)

        class_type = self.to_numpy(class_type).astype(np.int32)
        class_prob = self.to_numpy(class_prob).astype(np.float32)

        return int(class_type), np.max(class_prob)

    def split_np_label(self, image_mask, coords_info):
        split_list = []
        image_mask_pad, pad_info = self.vol_padding_by_cube(image_mask)

        for coord in coords_info:
            label = coord.label
            x, y, z = coord.coords.astype(np.int32)[:]
            new_x, new_y, new_z = x + \
                pad_info[0][0], y + pad_info[1][0], z + pad_info[2][0]

            binary_mask = (image_mask_pad == label).astype(np.int32)

            x_start, x_end = new_x - \
                self.cube_size[0]//2, new_x+self.cube_size[0]//2
            y_start, y_end = new_y - \
                self.cube_size[1]//2, new_y+self.cube_size[1]//2
            z_start, z_end = new_z - \
                self.cube_size[2]//2, new_z+self.cube_size[2]//2

            crop_image = binary_mask[x_start:x_end, y_start:y_end,
                                     z_start:z_end]

            split_list.append({"image": crop_image, "label": label})

        return split_list

    def split_np_img_label(self, image, image_mask, coords_info, stack=False):
        split_list = []
        image_mask_pad, pad_info = self.vol_padding_by_cube(image_mask)
        image_pad, _ = self.vol_padding_by_cube(image)

        for coord in coords_info:
            label = coord.label
            x, y, z = coord.coords.astype(np.int32)[:]
            new_x, new_y, new_z = x + \
                pad_info[0][0], y + pad_info[1][0], z + pad_info[2][0]

            binary_mask = (image_mask_pad == label).astype(np.int32)

            x_start, x_end = new_x - \
                self.cube_size[0]//2, new_x+self.cube_size[0]//2
            y_start, y_end = new_y - \
                self.cube_size[1]//2, new_y+self.cube_size[1]//2
            z_start, z_end = new_z - \
                self.cube_size[2]//2, new_z+self.cube_size[2]//2

            crop_msk = binary_mask[x_start:x_end, y_start:y_end, z_start:z_end]

            crop_img = image_pad[x_start:x_end, y_start:y_end, z_start:z_end]

            out = crop_img * crop_msk
            out[out == 0] = crop_img.min()

            if stack:
                out = np.stack([out, crop_msk], axis=0)

            split_list.append({"image": out, "label": label})

        return split_list

    @torch.no_grad()
    def inference_mask(self, inference_image_mask=None, coords_info=None):

        self.display_iter = 1

        vol_input = inference_image_mask

        strat_time = time.time()
        print("\nvertebra_fracture inference start...")

        split_list = self.split_np_label(vol_input, coords_info)
        vertebra_info_list = []

        for value in split_list:
            label = value["label"]
            vertebra_image = value["image"]

            class_type, class_prob = self.inference_image_type(vertebra_image,
                                                               stack=False)

            vertebra_info_list.append({
                "label": label,
                "class_type": class_type,
                "class_prob": class_prob
            })

        end_time = time.time()
        print(
            f"vertebra_fracture inference done..., elapsed time {(end_time-strat_time):.2f} secs"
        )

        return vertebra_info_list

    @torch.no_grad()
    def inference_img(self,
                      inference_image=None,
                      inference_image_mask=None,
                      coords_info=None,
                      stack=True):

        self.display_iter = 1

        image = inference_image
        mask = inference_image_mask

        strat_time = time.time()
        print("\nvertebra_fracture inference start...")

        split_list = self.split_np_img_label(image, mask, coords_info, stack)
        vertebra_info_list = []

        for value in split_list:
            label = value["label"]
            vertebra_image = value["image"]

            class_type, class_prob = self.inference_image_type(
                vertebra_image, stack)

            vertebra_info_list.append({
                "label": label,
                "class_type": class_type,
                "class_prob": class_prob
            })

        end_time = time.time()
        print(
            f"vertebra_fracture inference done..., elapsed time {(end_time-strat_time):.2f} secs"
        )

        return vertebra_info_list


class VertebraLabelClassifier(InferenceBase):

    def __init__(self,
                 prob_thr=0.5,
                 model_file=None,
                 model_func=generate_resnet_model) -> None:
        super().__init__(model_file, model_func)

        self.cube_size = (128, 128, 128)
        self.prob_thr = prob_thr

    def load_model(self):
        for model_name, model_path in self.model_file.items():
            cur_model = self.model_func(
                n_input_channels=1,
                model_depth=50,
                n_classes=self.get_classifier_type(model_name))
            state_dict = torch.load(model_path)["state_dict"]
            cur_model.load_state_dict(state_dict)
            if torch.cuda.is_available():
                cur_model.to(torch.device("cuda"))
            cur_model.eval()
            self.model_dict[model_name] = cur_model
            print(
                f"Model {cur_model.__class__.__name__} {model_name} Load Key Finish..."
            )

    def get_classifier_type(self, classifier_type):
        if classifier_type == 'group':
            nclass = 3
        elif classifier_type == 'cervical':
            nclass = 7
        elif classifier_type == 'thoracic':
            nclass = 12
        elif classifier_type == 'lumbar':
            nclass = 5
        return nclass

    def inference_image_type(self, model, image):

        vol = self.to_tensor(image)
        vol = vol.view(1, 1, vol.shape[0], vol.shape[1], vol.shape[2])
        out = model(vol)
        pred_label = torch.argmax(out)
        pred_prob = torch.nn.functional.softmax(out, dim=1)

        pred_label = self.to_numpy(pred_label)
        pred_prob = self.to_numpy(pred_prob)
        return int(pred_label), np.max(pred_prob)

    @torch.no_grad()
    def inference(self, inference_image=None, coords_info=None):
        torch.cuda.empty_cache()
        self.display_iter = 1
        if isinstance(inference_image, str):
            vol_input = self.read_image(inference_image)
        else:
            vol_input = inference_image

        crop_image_list = self.crop_by_coords(vol_input, coords_info)
        pred_info = []
        new_coords = []

        for i, (image, coords) in enumerate(zip(crop_image_list, coords_info)):

            group_label, group_prob = self.inference_image_type(
                self.model_dict["group"], image)

            if group_label == 0:
                # [0 - 6] - [1 - 7]
                class_label, class_prob = self.inference_image_type(
                    self.model_dict["cervical"], image)
                class_label = class_label + 1

            elif group_label == 1:
                # [0 - 11] - [8 - 19]
                class_label, class_prob = self.inference_image_type(
                    self.model_dict["thoracic"], image)
                class_label = class_label + 8

            elif group_label == 2:
                # [0 - 5] - [20 - 25]
                class_label, class_prob = self.inference_image_type(
                    self.model_dict["lumbar"], image)
                class_label = class_label + 20

            pred_info.append(
                [class_label, class_prob, group_label, group_prob])

            if group_prob < self.prob_thr or class_prob < self.prob_thr:
                continue
            else:
                self.display_info(
                    f"labelclassifier inference ori_label {int(coords.label)} --> new_label {int(class_label)}",
                    step=1)
                coords.label = class_label
                new_coords.append(coords)

        return pred_info, new_coords