import os
import numpy as np
import glob
import json
from inference import *
from utils.landmark.visualization.landmark_visualization_matplotlib import LandmarkVisualizationMatplotlib
from utils.landmark.landmark_statistics import LandmarkStatistics
from utils.segmentation.metrics import DiceMetric, SurfaceDistanceMetric, HausdorffDistanceMetric, PrecisionMetric, RecallMetric
from utils.segmentation.segmentation_statistics import SegmentationStatistics
from collections import OrderedDict
from utils.io.text import *
import pandas as pd


def remove_empty_folder(path):
    for base in os.listdir(path):
        folder = os.path.join(path, base)
        if os.path.isdir(folder) and len(os.listdir(folder)) == 0:
            print(f"remove folder --> {folder}")
            os.rmdir(folder)


def read_dice(path):
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "idv_muti_label_dice" in line:
                dice = float(line.split(" ")[-1])
                return dice


def is_continuous_list(list, step=1):
    prev = None
    for current in list:
        if prev is not None:
            if step != (current - prev):
                return False
        prev = current
    return True

def count_miss_point(pred_label, gt_label):
    sum = 0 
    for i in gt_label:
        if i not in pred_label:
            sum += 1
    return sum

def check_label():
    res_folder = "inference_result/Verse2020/postproc"
    verse_folder = "test_data/Verse2020"
    ids = os.listdir(res_folder)
    for id in ids:
        if os.path.isdir(os.path.join(res_folder, id)):

            pred_jsonfile = glob.glob(
                os.path.join(res_folder, id, "*.json"))[0]
            gt_jsonfile = glob.glob(os.path.join(
                verse_folder, id, "*.json"))[0]

            pred_info = read_annotations_from_json_file(pred_jsonfile)
            gt_info = read_annotations_from_json_file(gt_jsonfile)

            pred_info, gt_info, pred_label, gt_label = get_intersection(
                pred_info, gt_info)

            if pred_label != gt_label:
                print(f"image --> {id}")
            if not is_continuous_list(pred_label) or not is_continuous_list(gt_label):
                print(f"image --> {id}")


def check_dice():
    res_folder = "inference_result/Verse2020/without_postproc"
    ids = os.listdir(res_folder)

    dice_list = []
    falid_list = []

    for id in ids:
        if os.path.isdir(os.path.join(res_folder, id)):
            logfile = glob.glob(os.path.join(res_folder, id, "logs.txt"))[0]
            dice = read_dice(logfile)
            if dice < 0.9:
                falid_list.append(os.path.dirname(logfile))
            dice_list.append(dice)

    print(f"dice mean = {np.array(dice_list).mean()}")
    print(f"dice std = {np.array(dice_list).std()}")
    print(f"falid_list = {len(falid_list)} --> {falid_list}")


def read_annotations_from_json_file(file):

    with open(file, 'r') as f:
        data = f.read()
        anno = json.loads(data)

    locs = []

    for i in range(len(anno)):
        label = int(anno[i]['label'])
        x = float(anno[i]['X'])
        y = float(anno[i]['Y'])
        z = float(anno[i]['Z'])
        locs.append(Landmark(coords=np.array([x, y, z]),
                             is_valid=True, label=label, scale=1, value=0))

    return locs


def get_label(coords_info):
    labels = []
    for i in coords_info:
        labels.append(i.label)
    return labels


def get_intersection(pred_label, gt_label):
    intersection = list(set(pred_label) & set(gt_label))
    intersection = sorted(intersection, key=lambda x: x.label)
    a = intersection[0]
    b = intersection[-1]
    pred_label_i = pred_label[pred_label.index(a):pred_label.index(b)+1]
    gt_label_i = gt_label[gt_label.index(a):gt_label.index(b)+1]
    return pred_label_i, gt_label_i, get_label(pred_label_i), get_label(gt_label_i)


def write_list_to_file(save_path, save_list):
    if len(save_list) == 0:
        return
    with open(save_path, 'w') as file:
        for item in save_list:
            file.write(str(item) + '\n')


def eval_result(image_id, pred_folder, gt_floder, save_folder, vis, landmark_statistics, segmentation_statistics):

    landmark_statistics.reset()
    segmentation_statistics.reset()

    pred_label_path = glob.glob(os.path.join(pred_folder, "*.json"))[0]
    gt_label_path = glob.glob(os.path.join(gt_floder, "*.json"))[0]

    pred_label_info = read_annotations_from_json_file(pred_label_path)
    gt_label_info = read_annotations_from_json_file(gt_label_path)

    for path in glob.glob(os.path.join(pred_folder, "*.nii.gz")):
        if "_seg" in path:
            pred_seg_path = path

    for path in glob.glob(os.path.join(gt_floder, "*.nii.gz")):
        if "_seg" in path:
            gt_seg_path = path
        else:
            gt_image_path = path

    pred_seg = resample_to_spacing(sitk.ReadImage(
        pred_seg_path, sitk.sitkInt32), [1.0] * 3)
    gt_seg = resample_to_spacing(sitk.ReadImage(
        gt_seg_path, sitk.sitkInt32), [1.0] * 3)
    gt_image = resample_to_spacing(sitk.ReadImage(
        gt_image_path, sitk.sitkInt32), [1.0] * 3)

    pred_info, gt_info, pred_label, gt_label = get_intersection(
        pred_label_info, gt_label_info)
    
    print(f"gt_label = {gt_label}")
    print(f"pred_label = {pred_label}")
    #assert pred_label == gt_label
    
    if len(pred_label) == 0 or len(gt_label) == 0:
        return
    
    print("\nvisualize_landmark...")
    vis.visualize_landmark_projections(
        gt_image, gt_info, filename=os.path.join(save_folder, "gt_landmark.png"))
    vis.visualize_landmark_projections(
        gt_image, pred_info, filename=os.path.join(save_folder, "pred_landmark.png"))
    vis.visualize_prediction_groundtruth_projections(
        gt_image, pred_info, gt_info, filename=os.path.join(save_folder, "gt_pred_landmark.png"))
    print("\neval_landmark...")
    landmark_statistics.add_landmarks(image_id, pred_info, gt_info)
    overview_string = landmark_statistics.get_overview_string(
        [1, 2, 2.5, 3, 4, 10, 20], 10, 20.0)
    print(overview_string)
    save_string_txt(overview_string, os.path.join(
        save_folder, 'landmark_eval.txt'))
    print("\neval_segmentation...")
    segmentation_statistics.add_labels(image_id, pred_seg, gt_seg, gt_label)
    segmentation_statistics.finalize(save_folder)


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_mean_segment_eval(eval_path):
    ids = os.listdir(eval_path)
    dice_list = []
    hau_list = []
    precision_list = []
    recall_list = []
    for id in ids:
        dice = glob.glob(os.path.join(eval_path, id, "dice*.csv"))[0]
        hau = glob.glob(os.path.join(eval_path, id, "hau*.csv"))[0]
        precision = glob.glob(os.path.join(eval_path, id, "preci*.csv"))[0]
        recall = glob.glob(os.path.join(eval_path, id, "recall*.csv"))[0]

        dice_list.append(pd.read_csv(dice)["mean"][0])
        hau_list.append(pd.read_csv(hau)["mean"][0])
        precision_list.append(pd.read_csv(precision)["mean"][0])
        recall_list.append(pd.read_csv(recall)["mean"][0])

    return dice_list, hau_list, precision_list, recall_list


def get_mean_landmark_eval(eval_path):
    ids = os.listdir(eval_path)
    pe_mean_list = []
    pe_std_list = []
    pe_median_list = []

    ipe_mean_list = []
    ipe_std_list = []
    ipe_median_list = []

    for id in ids:
        lines = []
        landmark_path = glob.glob(os.path.join(
            eval_path, id, "landmark*.txt"))[0]
        with open(landmark_path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                lines.append(line)

        pe_mean = float(lines[1].split(":")[-1].split("\n")[0])
        pe_std = float(lines[2].split(":")[-1].split("\n")[0])
        pe_median = float(lines[3].split(":")[-1].split("\n")[0])

        ipe_mean = float(lines[5].split(":")[-1].split("\n")[0])
        ipe_std = float(lines[6].split(":")[-1].split("\n")[0])
        ipe_median = float(lines[7].split(":")[-1].split("\n")[0])

        pe_mean_list.append(pe_mean)
        pe_std_list.append(pe_std)
        pe_median_list.append(pe_median)

        ipe_mean_list.append(ipe_mean)
        ipe_std_list.append(ipe_std)
        ipe_median_list.append(ipe_median)

    return pe_mean_list, pe_std_list, pe_median_list, ipe_mean_list, ipe_std_list, ipe_median_list

def eval(gt_base_folder, pred_base_folder, output_folder):
    segmentation_statistics = SegmentationStatistics(
        metrics=OrderedDict([('dice', DiceMetric()),
                             (('hau_distance'), HausdorffDistanceMetric(
                             )),
                             ('precision', PrecisionMetric(
                             )),
                             ('recall', RecallMetric(
                             )),
                             ])
    )
    vis = LandmarkVisualizationMatplotlib(dim=3,
                                          annotations=dict([(i, f'C{i + 1}') for i in range(7)] +        # 0-6: C1-C7
                                                           # 7-18: T1-12
                                                           [(i, f'T{i - 6}') for i in range(7, 19)] +
                                                           # 19-24: L1-6
                                                           [(i, f'L{i - 18}') for i in range(19, 25)] +
                                                           [(25, 'T13')]))                               # 25: T13
    landmark_statistics = LandmarkStatistics()

    check_dir(output_folder)
    gt = os.listdir(gt_base_folder)
    pred = os.listdir(pred_base_folder)
    for index, (gt_folder, pred_folder) in enumerate(zip(gt, pred)):

        assert gt_folder == pred_folder

        print(
            f"\neval image landmark and segment on {gt_folder}, {index+1}/{len(gt)}...")

        output_path = os.path.join(output_folder, gt_folder)
        check_dir(output_path)

        gt_folder = os.path.join(gt_base_folder, gt_folder)
        pred_folder = os.path.join(pred_base_folder, pred_folder)

        eval_result(image_id=gt_folder,
                    pred_folder=pred_folder,
                    gt_floder=gt_folder,
                    save_folder=output_path,
                    landmark_statistics=landmark_statistics,
                    vis=vis,
                    segmentation_statistics=segmentation_statistics)

if __name__ == "__main__":

    gt_base_folder = "test_data"
    pred_base_folder = "inference_result"
    output_folder = "eval_result"
    
    eval(gt_base_folder, pred_base_folder, output_folder)