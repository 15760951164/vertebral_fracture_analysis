import os
import torch
import glob
import sys
sys.path.append("./")
from models.UNet import UNet3D_ResidualSE, ResidualUNet3D, UNet3D
from train.loss import DiceL2Loss
from torch.utils.data import DataLoader
import argparse
from train.train_base import TrainLoop, default_data_extraction_func
from utils.sitk_np import sitk_to_npimage, npimage_to_sitk
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
from skimage import filters
import random

class spine_dataset(Dataset):

    def __init__(self, data_dir):
        super(spine_dataset, self).__init__

        self.filelist_img = []
        self.filelist_msk = []

        train_ids = os.listdir(data_dir)

        for vol_id in train_ids:
            id_folder = os.path.join(data_dir, vol_id)
            img_dir = os.path.join(id_folder, "img")
            msk_dir = os.path.join(id_folder, "msk")

            img_files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
            msk_files = sorted(glob.glob(os.path.join(msk_dir, "*.nii.gz")))

            assert len(img_files) == len(msk_files)

            for img_file, msk_file in zip(img_files, msk_files):
                self.filelist_img.append(img_file)
                self.filelist_msk.append(msk_file)

    def __getitem__(self, index):

        assert len(self.filelist_img) == len(self.filelist_msk)

        image_id = os.path.basename(self.filelist_img[index])[:-len(".nii.gz")]

        img_cube = sitk.ReadImage(self.filelist_img[index])
        msk_cube = sitk.ReadImage(self.filelist_msk[index])

        img_cube_array = sitk_to_npimage(img_cube)
        msk_cube_array = sitk_to_npimage(msk_cube)

        img_cube_array = torch.from_numpy(
            img_cube_array).to(dtype=torch.float32)
        msk_cube_array = torch.from_numpy(
            msk_cube_array).to(dtype=torch.float32)

        self.msk_cube = msk_cube_array.view(1, msk_cube_array.size(0),
                                            msk_cube_array.size(1),
                                            msk_cube_array.size(2))
        self.img_cube = img_cube_array.view(1, img_cube_array.size(0),
                                            img_cube_array.size(1),
                                            img_cube_array.size(2))

        return self.img_cube, self.msk_cube, image_id

    def __len__(self):
        return len(self.filelist_img)


class spine_train(TrainLoop):

    def __init__(self,
                 model,
                 loss_function,
                 train_dataloader,
                 lr=1e-3,
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
        debug_dir = "/mnt/e/wyh/vertbrae/model_weight/spine/test_out"
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

                if i == debug_index:
                    seg = predicted_label[0][0].cpu().numpy()
                    sitk.WriteImage(npimage_to_sitk(seg),
                                    os.path.join(debug_dir, f"{self.current_iter}_spine_segment.nii.gz"))
                    
                print('test epoch = {}, iter = {}/{}, loss = {}'.format(
                    self.current_iter, i, len(self.test_dataloader), loss))

        return epoch_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        'spine or vertebra segmentor train script')

    parser.add_argument('--save_dir',
                        type=str,
                        default="/mnt/e/wyh/vertbrae/model_weight/spine")

    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument(
        '--train_dataset_dir',
        type=str,
        default="/mnt/e/wyh/vertbrae/train/data/spine_segmentor/train")

    parser.add_argument(
        '--test_dataset_dir',
        type=str,
        default="/mnt/e/wyh/vertbrae/train/data/spine_segmentor/test")

    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--n_epoch', type=int, default=35)

    parser.add_argument('--workers', type=int, default=14)

    args = parser.parse_args()

    train_dataset = spine_dataset(args.train_dataset_dir)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  num_workers=args.workers,
                                  drop_last=False,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    test_dataset = spine_dataset(args.test_dataset_dir)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 num_workers=args.workers,
                                 drop_last=False,
                                 batch_size=args.batch_size,
                                 shuffle=True)

    model = UNet3D_ResidualSE(in_channels=1, out_channels=1, f_maps=32, layer_order="cbr", repeats=1,
                              final_activation="sigmoid", conv_kernel_size=3, conv_padding=1, use_attn=False, num_levels=5)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=1e-4)

    loss = DiceL2Loss(lambda_l2=20.0)

    main_train_loop = spine_train(model=model,
                                  loss_function=loss,
                                  train_dataloader=train_dataloader,
                                  lr=args.lr,
                                  test_dataloader=test_dataloader,
                                  optimizer=optimizer,
                                  model_load_path=os.path.join(args.save_dir, "min_loss_weight.pth"),
                                  model_save_path=args.save_dir,
                                  max_iter=args.n_epoch)

    main_train_loop.run()
