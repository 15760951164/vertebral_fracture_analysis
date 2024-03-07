
import numpy as np
import os
import sys
sys.path.append("./")
import torch
import glob
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import numpy as np
from utils.sitk_np import sitk_to_npimage, npimage_to_sitk
from train.train_base import TrainLoop, default_data_extraction_func
from models.ResNet import generate_resnet_model
import argparse
import json

class group_dataset(Dataset):

    def __init__(self, data_dir):
        super(group_dataset, self).__init__

        self.filelist = []
        self.labellist = []

        train_ids = os.listdir(data_dir)

        for vol_id in train_ids:
            id_folder = os.path.join(data_dir, vol_id)

            msk_files = glob.glob(os.path.join(id_folder, "*.nii.gz"))

            for file in msk_files:
                bone_label = int(file.split("bone")[-1].split("_")[0])
                self.filelist.append(file)
                self.labellist.append(bone_label)

    def __getitem__(self, index):
        assert len(self.filelist) == len(self.labellist)

        image_id = os.path.basename(self.filelist[index])
        msk_cube = sitk_to_npimage(sitk.ReadImage(self.filelist[index]))
        out = torch.from_numpy(np.expand_dims(msk_cube,
                                              axis=0)).to(torch.float32)

        label = self.labellist[index]
        if label <= 7:
            label = 0
        elif label <= 19 and label >= 8:
            label = 1
        elif label >= 20 and label != 28:
            label = 2
        else:
            label = 1

        self.msk_cube = out
        self.label = label

        return self.msk_cube, self.label, image_id

    def __len__(self):
        return len(self.filelist)


class cervical_dataset(Dataset):

    def __init__(self, data_dir):
        super(cervical_dataset, self).__init__

        train_ids = os.listdir(data_dir)

        self.filelist = []
        self.labellist = []

        for vol_id in train_ids:
            id_folder = os.path.join(data_dir, vol_id)

            msk_files = glob.glob(os.path.join(id_folder, "*.nii.gz"))

            for file in msk_files:
                bone_label = int(file.split("bone")[-1].split("_")[0])
                if bone_label > 7:
                    continue
                self.filelist.append(file)
                self.labellist.append(bone_label)

    def __getitem__(self, index):
        assert len(self.filelist) == len(self.labellist)

        image_id = os.path.basename(self.filelist[index])
        msk_cube = sitk_to_npimage(sitk.ReadImage(self.filelist[index]))
        out = torch.from_numpy(np.expand_dims(msk_cube,
                                              axis=0)).to(torch.float32)

        label = self.labellist[index] - 1

        self.msk_cube = out

        self.label = label

        return self.msk_cube, self.label, image_id

    def __len__(self):
        return len(self.filelist)


class thoracic_dataset(Dataset):

    def __init__(self, data_dir):
        super(thoracic_dataset, self).__init__

        train_ids = os.listdir(data_dir)

        self.filelist = []
        self.labellist = []

        for vol_id in train_ids:
            id_folder = os.path.join(data_dir, vol_id)

            msk_files = glob.glob(os.path.join(id_folder, "*.nii.gz"))

            for file in msk_files:
                bone_label = int(file.split("bone")[-1].split("_")[0])
                if bone_label == 28:
                    self.filelist.append(file)
                    self.labellist.append(bone_label)
                elif bone_label < 8 or bone_label > 19:
                    continue
                else:
                    self.filelist.append(file)
                    self.labellist.append(bone_label)

    def __getitem__(self, index):
        assert len(self.filelist) == len(self.labellist)

        image_id = os.path.basename(self.filelist[index])
        msk_cube = sitk_to_npimage(sitk.ReadImage(self.filelist[index]))
        out = torch.from_numpy(np.expand_dims(msk_cube,
                                              axis=0)).to(torch.float32)

        label = self.labellist[index]

        if label == 28:
            label = 11
        else:
            label -= 8

        self.msk_cube = out
        self.label = label

        return self.msk_cube, self.label, image_id

    def __len__(self):
        return len(self.filelist)


class lumbar_dataset(Dataset):

    def __init__(self, data_dir):
        super(lumbar_dataset, self).__init__

        train_ids = os.listdir(data_dir)

        self.filelist = []
        self.labellist = []

        for vol_id in train_ids:
            id_folder = os.path.join(data_dir, vol_id)

            msk_files = glob.glob(os.path.join(id_folder, "*.nii.gz"))

            for file in msk_files:
                bone_label = int(file.split("bone")[-1].split("_")[0])
                if bone_label < 20 or bone_label == 28:
                    continue
                self.filelist.append(file)
                self.labellist.append(bone_label)

    def __getitem__(self, index):
        assert len(self.filelist) == len(self.labellist)

        image_id = os.path.basename(self.filelist[index])
        msk_cube = sitk_to_npimage(sitk.ReadImage(self.filelist[index]))
        out = torch.from_numpy(np.expand_dims(msk_cube,
                                              axis=0)).to(torch.float32)

        label = self.labellist[index]

        if label == 25:
            label = 25
        else:
            label -= 20

        self.msk_cube = out
        self.label = label

        return self.msk_cube, self.label, image_id

    def __len__(self):
        return len(self.filelist)


class fracture_dataset(Dataset):

    def __init__(self, data_dir):
        super(fracture_dataset, self).__init__

        self.filelist = []

        train_ids = os.listdir(data_dir)

        for vol_id in train_ids:

            if vol_id == "fracture":
                label = 0
                fracture_files = glob.glob(
                    os.path.join(data_dir, vol_id, "*.npz"))
                for file in fracture_files:
                    self.filelist.append({"label": label, "image_path": file})

            elif vol_id == "normal":
                label = 1
                normal_files = glob.glob(
                    os.path.join(data_dir, vol_id, "*.npz"))
                for file in normal_files:
                    self.filelist.append({"label": label, "image_path": file})
            else:
                label = -1

            assert label != -1

    def __getitem__(self, index):

        image_path = self.filelist[index]["image_path"]
        label = self.filelist[index]["label"]

        img_cube_array = np.load(image_path)["arr_0"]

        img_cube_array = torch.from_numpy(img_cube_array).to(
            dtype=torch.float32)
        self.img_cube = img_cube_array

        self.label = torch.tensor(label).to(torch.long)

        return self.img_cube, self.label, str(image_path)

    def __len__(self):
        return len(self.filelist)


class classifier_train(TrainLoop):

    def __init__(self,
                 model,
                 loss_function,
                 train_dataloader,
                 lr=0.001,
                 data_extraction_func=default_data_extraction_func,
                 test_dataloader=None,
                 optimizer=None,
                 model_save_path=None,
                 model_load_path=None,
                 max_iter=150,
                 checkpoint_iter=5,
                 weight_init_type="xavier") -> None:
        super().__init__(model, loss_function, train_dataloader, lr,
                         data_extraction_func, test_dataloader, optimizer,
                         model_save_path, model_load_path, max_iter,
                         checkpoint_iter, weight_init_type)
        



def get_class_weight(group_level, label_count_path):
    
    with open(label_count_path, 'r') as f:
        data = f.read()
        vert_count = json.loads(data)

    if group_level == 'group':
        arr = np.zeros(3)
        for l in range(1, 8):
            arr[0] += vert_count[str(l)]
        for l in range(8, 20):
            arr[1] += vert_count[str(l)]
        for l in range(20, 26):
            arr[2] += vert_count[str(l)]
        weight =  1.0 / arr

    elif group_level == 'cervical':
        arr = np.zeros(7)
        for l in range(1, 8):
            arr[l - 1] += vert_count[str(l)]
        weight =  1.0 / arr

    elif group_level == 'thoracic':
        arr = np.zeros(12)
        for l in range(8, 20):
            arr[l - 8] += vert_count[str(l)]
        weight =  1.0 / arr

    elif group_level == 'lumbar':
        arr = np.zeros(5)
        for l in range(20, 26):
            arr[l - 20] += vert_count[str(l)]
        weight =  1.0 / arr
    
    return torch.from_numpy(weight).to(torch.float32).cuda()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--save_dir',
        type=str,
        default="/mnt/e/wyh/vertbrae/model_weight/classifier",
    )

    parser.add_argument('--classify_level',
                        type=str,
                        default="cervical",
                        help='group | cervical | thoracic | lumbar | fracture')

    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--train_dataset_dir',
                        type=str,
                        default="/mnt/e/wyh/vertbrae/train/data/fracture/train")

    parser.add_argument('--test_dataset_dir',
                        type=str,
                        default="/mnt/e/wyh/vertbrae/train/data/fracture/test")

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_epoch', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)

    args = parser.parse_args()
    label_count_path = "classifier_num_of_each_label.json"

    if args.classify_level == 'group':
        model = generate_resnet_model(n_input_channels=1,
                                      model_depth=50,
                                      n_classes=3)
        class_weights = get_class_weight('group', label_count_path)
        train_dataset = group_dataset(data_dir=args.train_dataset_dir)
        test_dataset = group_dataset(data_dir=args.test_dataset_dir)

    elif args.classify_level == 'cervical':  # 颈椎
        model = generate_resnet_model(n_input_channels=1,
                                      model_depth=50,
                                      n_classes=7)
        class_weights = get_class_weight('cervical', label_count_path)
        train_dataset = cervical_dataset(data_dir=args.train_dataset_dir)
        test_dataset = cervical_dataset(data_dir=args.test_dataset_dir)

    elif args.classify_level == 'thoracic':  # 腰椎

        model = generate_resnet_model(n_input_channels=1,
                                      model_depth=50,
                                      n_classes=12)
        class_weights = get_class_weight('thoracic', label_count_path)
        train_dataset = thoracic_dataset(data_dir=args.train_dataset_dir)
        test_dataset = thoracic_dataset(data_dir=args.test_dataset_dir)

    elif args.classify_level == 'lumbar':  # 胸椎

        model = generate_resnet_model(n_input_channels=1,
                                      model_depth=50,
                                      n_classes=5)
        class_weights = get_class_weight('lumbar', label_count_path)
        train_dataset = lumbar_dataset(data_dir=args.train_dataset_dir)
        test_dataset = lumbar_dataset(data_dir=args.test_dataset_dir)

    elif args.classify_level == "fracture":

        class_weights = None
        model = generate_resnet_model(n_input_channels=2,
                                      model_depth=50,
                                      n_classes=2)
        train_dataset = fracture_dataset(data_dir=args.train_dataset_dir)
        test_dataset = fracture_dataset(data_dir=args.test_dataset_dir)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  num_workers=args.workers,
                                  drop_last=False,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 num_workers=args.workers,
                                 drop_last=False,
                                 batch_size=args.batch_size,
                                 shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=1e-4)

    loss = torch.nn.CrossEntropyLoss(weight=class_weights)

    main_train = classifier_train(model=model,
                                  loss_function=loss,
                                  train_dataloader=train_dataloader,
                                  lr=args.lr,
                                  test_dataloader=test_dataloader,
                                  optimizer=optimizer,
                                  model_load_path=None,
                                  model_save_path=os.path.join(args.save_dir, args.classify_level),
                                  max_iter=args.n_epoch)
    
    main_train.run()