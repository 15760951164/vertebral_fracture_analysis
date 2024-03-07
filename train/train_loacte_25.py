import sys
sys.path.append("./")
from utils.heatmap import generator_heatmaps_by_msk
import glob
import torch
import os
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
from utils.sitk_np import sitk_to_npimage, npimage_to_sitk
from train.train_base import TrainLoop, default_data_extraction_func
import argparse
from torch.utils.data import DataLoader
from models.UNet import UNet3D_ResidualSE, ResidualUNet3D, UNet3D
from models.SCN import SCN
from models import SCN
from models import SCN_Modify
import random
from train.loss import lambda_l2_loss

class loacte_dataset(Dataset):

    def __init__(self, data_dir, stack=False):
        super(loacte_dataset, self).__init__

        self.filelist_img = []
        self.filelist_heatmap = []
        self.stack = stack

        train_ids = os.listdir(data_dir)

        for vol_id in train_ids:
            id_folder = os.path.join(data_dir, vol_id)
            img_dir = os.path.join(id_folder, "img")
            heatmap_dir = os.path.join(id_folder, "mskheatmap")

            img_files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
            img_base_name = ".nii.gz"
            if self.stack:
                heatmap_files = sorted(
                    glob.glob(os.path.join(heatmap_dir, "*.npz")))
                heatmap_base_name = ".npz"
            else:
                heatmap_files = sorted(
                    glob.glob(os.path.join(heatmap_dir, "*.nii.gz")))
                heatmap_base_name = ".nii.gz"

            assert len(img_files) == len(heatmap_files)

            for img_file, heatmap_file in zip(img_files, heatmap_files):
                assert os.path.basename(
                    img_file)[:-len(img_base_name)] == os.path.basename(
                        heatmap_file)[:-len(heatmap_base_name)]
                self.filelist_img.append(img_file)
                self.filelist_heatmap.append(heatmap_file)

    def __getitem__(self, index):

        assert len(self.filelist_img) == len(self.filelist_heatmap)

        img_cube = sitk.ReadImage(self.filelist_img[index])
        img_cube_array = sitk_to_npimage(img_cube)
        img_cube_array = torch.from_numpy(img_cube_array).to(
            dtype=torch.float32)

        if self.stack:
            heatmap_cube_array = np.load(self.filelist_heatmap[index])["arr_0"]
        else:
            heatmap_cube_array = sitk_to_npimage(
                sitk.ReadImage(self.filelist_heatmap[index]))
        heatmap_cube_array = torch.from_numpy(heatmap_cube_array).to(
            dtype=torch.float32)

        self.img_cube = img_cube_array.unsqueeze(dim=0)

        if self.stack:
            self.heatmap_cube = heatmap_cube_array
        else:
            self.heatmap_cube = heatmap_cube_array.unsqueeze(dim=0)

        return self.img_cube, self.heatmap_cube, os.path.basename(
            self.filelist_img[index])[:-len(".nii.gz")]

    def __len__(self):
        return len(self.filelist_img)


class loacte_dataset_msk(Dataset):

    def __init__(self, data_dir, stack):
        super(loacte_dataset_msk, self).__init__

        self.filelist_img = []
        self.filelist_msk = []

        train_ids = os.listdir(data_dir)

        for vol_id in train_ids:
            id_folder = os.path.join(data_dir, vol_id)
            img_dir = os.path.join(id_folder, "img")
            msk = os.path.join(id_folder, "msk")

            img_files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
            msk_files = sorted(glob.glob(os.path.join(msk, "*.nii.gz")))
            img_base_name = ".nii.gz"
            heatmap_base_name = ".nii.gz"

            assert len(img_files) == len(msk_files)

            for img_file, msk_file in zip(img_files, msk_files):
                assert os.path.basename(
                    img_file)[:-len(img_base_name)] == os.path.basename(
                        msk_file)[:-len(heatmap_base_name)]
                self.filelist_img.append(img_file)
                self.filelist_msk.append(msk_file)

    def __getitem__(self, index):

        assert len(self.filelist_img) == len(self.filelist_msk)

        img_cube = sitk.ReadImage(self.filelist_img[index])
        img_cube_array = sitk_to_npimage(img_cube)
        img_cube_array = torch.from_numpy(img_cube_array).to(
            dtype=torch.float32)

        heatmap_cube_array = generator_heatmaps_by_msk(
            msk=sitk_to_npimage(sitk.ReadImage(self.filelist_msk[index])))

        heatmap_cube_array = torch.from_numpy(heatmap_cube_array).to(
            dtype=torch.float32)

        self.img_cube = img_cube_array.unsqueeze(dim=0)

        self.heatmap_cube = heatmap_cube_array

        return self.img_cube, self.heatmap_cube, os.path.basename(
            self.filelist_img[index])[:-len(".nii.gz")]

    def __len__(self):
        return len(self.filelist_img)


class locate_train(TrainLoop):

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
        
    @torch.no_grad()
    def test_step(self):

        epoch_loss = 0
        self.model.eval()
        debug_dir = "/mnt/e/wyh/vertbrae/model_weight/idv_locate_25_channel/test_out"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        debug_index = random.randint(1, len(self.test_dataloader)-1)

        if self.test_dataloader is not None:

            for i, batch in enumerate(self.test_dataloader):
                # model inputs
                input, label = self.data_extraction_func(batch)
                input = input.cuda()
                label = label.cuda()

                predicted_label = self.model(input)

                loss = self.loss_function(predicted_label, label)

                epoch_loss += loss.item()
                
                print('test epoch = {}, iter = {}/{}, loss = {}'.format(
                    self.current_iter, i, len(self.test_dataloader)-1, loss))

                if i == debug_index:

                    vol_output = predicted_label[0].cpu().numpy()
                    heatmaps = np.zeros(
                        (vol_output.shape[1], vol_output.shape[2], vol_output.shape[3]), dtype=np.float32)
                    debug_dir = os.path.join(
                        debug_dir, f"epoch_{self.current_iter}")

                    if not os.path.exists(debug_dir):
                        os.makedirs(debug_dir)

                    for i in range(vol_output.shape[0]):
                        heatmaps += vol_output[i]

                    sitk.WriteImage(npimage_to_sitk(heatmaps),
                                    os.path.join(debug_dir, "all_heatmap_pred.nii.gz"))

        return epoch_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        'spine or vertebra segmentor train script')

    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096"

    parser.add_argument('--save_dir',
                        type=str,
                        default="/mnt/e/wyh/vertbrae/model_weight/idv_locate_25_channel")

    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument(
        '--train_dataset_dir',
        type=str,
        default="/mnt/e/wyh/vertbrae/train/data/idv_locate_25_channel/train")

    parser.add_argument(
        '--test_dataset_dir',
        type=str,
        default="/mnt/e/wyh/vertbrae/train/data/idv_locate_25_channel/test")

    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--n_epoch', type=int, default=200)

    parser.add_argument('--workers', type=int, default=14)
    
    parser.add_argument('--lambda_l2', type=int, default=450)

    args = parser.parse_args()
    print(args)

    train_dataset = loacte_dataset(args.train_dataset_dir, stack=True)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  num_workers=args.workers,
                                  drop_last=False,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    test_dataset = loacte_dataset(args.test_dataset_dir, stack=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 num_workers=args.workers,
                                 drop_last=False,
                                 batch_size=args.batch_size,
                                 shuffle=True)

    model = SCN_Modify.SCN(in_channels=1, out_channels=25, f_maps=64, dropout_prob=0.3)
    # model = SCN_test.SCN(in_channels=1, out_channels=25, f_maps=96)
    # model = SCN_UNet(1, 26, filters=96)
    # model = UNet3D(in_channels=1, out_channels=25, f_maps=96, layer_order="cbl", repeats=1, 
    #                pool_type="max", final_activation=None, conv_kernel_size=3, 
    #                conv_padding=1, use_attn=False, num_levels=5)
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=1e-4)
    
    loss = lambda_l2_loss(lambda_l2=args.lambda_l2)

    main_train = locate_train(model=model,
                              loss_function=loss,
                              train_dataloader=train_dataloader,
                              lr=args.lr,
                              test_dataloader=test_dataloader,
                              optimizer=optimizer,
                              model_load_path=os.path.join(args.save_dir, "last_train_weight.pth"),
                              model_save_path=args.save_dir,
                              max_iter=args.n_epoch)
    main_train.run()