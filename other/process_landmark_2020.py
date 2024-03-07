import json
import os
from glob import glob
import SimpleITK as sitk
import numpy as np
import copy

base_floder = "data"
image_folder = "data//images"
save_floder = "data//images_reoriented"

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

for id in os.listdir(image_folder):
    
    files = glob(os.path.join(image_folder, id, '*.json'))
    print(files)
    
    for filename in sorted(files):
        filename_wo_folder = os.path.basename(filename)
        ext_length = len('_iso-ctd.json')
        filename_wo_folder_and_ext = filename_wo_folder[:-ext_length]
        image_id = filename_wo_folder_and_ext

        image_meta_data = sitk.ReadImage(
            os.path.join(base_floder, 'images_reoriented',
                        id, image_id + '.nii.gz'))
        
        spacing = np.array(image_meta_data.GetSpacing())
        origin = np.array(image_meta_data.GetOrigin())
        direction = np.array(image_meta_data.GetDirection()).reshape([3, 3])
        size = np.array(image_meta_data.GetSize())
        
        with open(filename, 'r') as f:
            # load json file
            json_data = json.load(f)
            for landmark in json_data:
                landmark_copy = copy.deepcopy(landmark)
                            
                landmark["X"] = size[0] * spacing[0] - float(landmark_copy['Z'])
                landmark["Y"] = float(landmark_copy['X'])
                landmark["Z"] = size[2] * spacing[2] - float(landmark_copy['Y'])
        
        output_path = os.path.join(save_floder, id, image_id + "_iso-ctd.json")
        with open(output_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
            