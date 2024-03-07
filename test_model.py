import numpy as np
from inference import Landmark
import json 


from inference import SpineSegmention

inference_image_path = "test_data/GL195/GL195_CT_ax.nii.gz"

model_file = "result/model_weight/spine/UNet3D_ResidualSE_00025.pth"

spine_seg = SpineSegmention(model_file=model_file)

binary_mask, roi_box = spine_seg.inference(
    inference_image=inference_image_path, save_path="test_out")

from inference import VertebraLocate_1_Channel

model_file = "result/model_weight/idv_locate_1_channel/SCN_00140.pth"

locate_single = VertebraLocate_1_Channel(model_file=model_file)

heatmap, coords_info = locate_single.inference(
    inference_image=inference_image_path,
    roi_box=roi_box,
    save_path="test_out")

for coord in coords_info:
    if coord.is_valid == True:
        vertebra_locate_logs = f"label = {coord.label}_coords = {coord.coords}"
        print(vertebra_locate_logs)

from inference import VertebraLabelClassifier

label_classifier = VertebraLabelClassifier(
    model_file={
        "group": "result/model_weight/classifier/group/ResNet_00050.pth",
        "cervical": "result/model_weight/classifier/cervical/ResNet_00110.pth",
        "thoracic": "result/model_weight/classifier/thoracic/ResNet_00100.pth",
        "lumbar": "result/model_weight/classifier/lumbar/ResNet_00100.pth"
    })

pred_info, coords_info = label_classifier.inference(binary_mask, coords_info)

for coord in coords_info:
    if coord.is_valid == True:
        vertebra_locate_logs = f"label = {coord.label}_coords = {coord.coords}"
        print(vertebra_locate_logs)

from inference import VertebraLocate_25_Channel

model_file = "result/model_weight/idv_locate_25_channel/SCN_00200.pth"

loacte_muti = VertebraLocate_25_Channel(model_file=model_file)

heatmap, coords_info = loacte_muti.inference(
    inference_image=inference_image_path,
    roi_box=roi_box,
    save_path="test_out")

for coord in coords_info:
    if coord.is_valid == True:
        vertebra_locate_logs = f"label = {coord.label}_coords = {coord.coords}"
        print(vertebra_locate_logs)

from inference import VertebraSegmention

json_file = "test_data/GL195/GL195_CT_ax_iso-ctd.json"

with open(json_file, 'r') as f:
    data = f.read()
    anno = json.loads(data)
    locs = []
    for i in range(len(anno)):
        label = int(anno[i]['label'])
        x = int(anno[i]['X'])
        y = int(anno[i]['Y'])
        z = int(anno[i]['Z'])
        locs.append(
            Landmark(coords=np.array([x, y, z]),
                     is_valid=True,
                     label=label,
                     scale=1,
                     value=0))

model_file = "result/model_weight/vertebra/UNet3D_ResidualSE_00100.pth"

idv_seg = VertebraSegmention(model_file=model_file)

muti_label_mask, vertebra_mask_list = idv_seg.inference(
    inference_image=inference_image_path,
    save_path="test_out",
    coord_info=locs,
    filter=True)