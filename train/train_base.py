import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.init as init
import time

def default_data_extraction_func(batch):
    return batch[0], batch[1]

class TrainLoop(object):

    def __init__(self,
                 model,
                 loss_function,
                 train_dataloader,
                 lr=1e-3,
                 data_extraction_func=None,
                 test_dataloader=None,
                 optimizer=None,
                 model_save_path=None,
                 model_load_path=None,
                 max_iter=150,
                 checkpoint_iter=5,
                 weight_init_type="xavier") -> None:

        self.model_save_path = model_save_path
        self.model_load_path = model_load_path
        self.model = model
        self.lr = lr
        self.min_test_loss = np.iinfo(np.int32).max
        self.loss_function = loss_function
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.optimizer = optimizer
        self.data_extraction_func = data_extraction_func

        self.max_iter = max_iter
        self.current_iter = 1
        self.checkpoint_iter = checkpoint_iter
        self.weight_init_type = weight_init_type
        self.log_dir = model_save_path
        self.model_name = model.__class__.__name__

        self.init_all()

    def init_all(self):
        
        print(self.model)
        
        self.init_model()
        
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.lr,
                                              weight_decay=1e-4)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.2, patience=5, min_lr=5e-7, cooldown=5, verbose=True)
        
        if self.model_load_path is not None:
            self.load_model()
        
            
    def load_model(self):

        if self.model_load_path is not None:
            
            checkpoint = torch.load(self.model_load_path, map_location='cpu')

            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])

            if self.optimizer is not None and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                
            if self.scheduler is not None and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                
            if 'current_iter' in checkpoint:
                self.current_iter = checkpoint['current_iter'] + 1
                
            if 'min_test_loss' in checkpoint:
                self.min_test_loss = checkpoint['min_test_loss']
                
            print(f'loading checkpoint {self.model_load_path} train utils, current_iter {self.current_iter}, current_test_loss {self.min_test_loss}')

    def save_full_model_info(self, model_name=None):

        if self.model_save_path is not None:
            
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)
            if model_name is None:
                model_name = f"{self.model_name}_{self.current_iter:05d}.pth"
            save_file_path = os.path.join(self.model_save_path, model_name)
            if hasattr(self.model, 'module'):
                model_state_dict = self.model.module.state_dict()
            else:
                model_state_dict = self.model.state_dict()
            save_states = {
                'current_iter': self.current_iter,
                'state_dict': model_state_dict,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'min_test_loss': self.min_test_loss
            }
            torch.save(save_states, save_file_path)
            
    def save_model(self, model_name=None):

        if self.model_save_path is not None:
            
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)
            if model_name is None:
                model_name = f"{self.model_name}_{self.current_iter:05d}"
            save_file_path = os.path.join(self.model_save_path, model_name + ".pth")
            if hasattr(self.model, 'module'):
                model_state_dict = self.model.module.state_dict()
            else:
                model_state_dict = self.model.state_dict()
            save_states = {
                'state_dict': model_state_dict,
            }
            torch.save(save_states, save_file_path)

    def init_model(self):

        self.init_gpu()
        self.init_weight()

    def init_weight(self):

        print('initialize network with method {}'.format(
            self.weight_init_type))

        for m in self.model.modules():
            if isinstance(m, nn.Conv3d):
                if self.weight_init_type == 'normal':
                    init.normal_(m.weight, mean=0.0, std=1.0)
                elif self.weight_init_type == 'uniform':
                    init.uniform_(m.weight, a=0.0, b=1.2)
                elif self.weight_init_type == 'xavier':
                    init.xavier_normal_(m.weight, gain=0.02)
                elif self.weight_init_type == 'kaiming':
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                elif self.weight_init_type == 'orthogonal':
                    init.orthogonal_(m.weight, gain=0.02)
                elif self.weight_init_type == 'ones':
                    init.ones_(m.weight)
                else:
                    raise NotImplementedError(
                        'initialzation method {} NOT implemented.'.format(
                            self.weight_init_type))
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            if isinstance(m, nn.BatchNorm3d):
                init.normal_(m.weight, mean=0.0, std=1.0)
                init.constant_(m.bias, 0.0)

    def init_gpu(self):

        gpu_ids = np.arange(torch.cuda.device_count()).tolist()
        if len(gpu_ids) == 1:
            self.model = self.model.cuda()
        else:
            print('multi gpus processing.')
            self.model.to(gpu_ids[0])
            self.model = torch.nn.DataParallel(self.model, gpu_ids)
            print('gpu ids: ', gpu_ids)

    def check_run(self):

        if self.model == None or self.lr == None or self.loss_function == None or self.train_dataloader == None\
            or self.test_dataloader == None or self.optimizer == None or self.data_extraction_func == None:
            return False

        return True

    def run(self):

        assert self.check_run()
        
        while self.current_iter <= self.max_iter:
            
            start_time = time.time()
            
            train_loss = self.train_step()
            
            test_loss = self.test_step()

            elapsed_time = time.time() - start_time

            self.scheduler.step(train_loss)

            if self.min_test_loss > test_loss:
                self.min_test_loss = test_loss
                self.save_full_model_info(model_name="min_loss_weight.pth")

            if self.current_iter % self.checkpoint_iter == 0:
                self.save_model()
            
            self.save_full_model_info(model_name="last_train_weight.pth")

            epoch_info = '%4d/%4d  train_loss = %.10f, test_loss = %.10f, train_avg_loss = %.10f, test_avg_loss = %.10f, time= %4.2fs ,lr = %.10f, min_loss = %.10f' %\
                    (self.current_iter, self.max_iter, train_loss, test_loss , train_loss / len(self.train_dataloader), test_loss / len(self.test_dataloader), \
                        elapsed_time, self.optimizer.param_groups[0]['lr'], self.min_test_loss)
            print(epoch_info, '\n')
            
            self.epoch_info_to_file(epoch_info)

            self.current_iter += 1

    def epoch_info_to_file(self, text):

        if self.log_dir is not None:

            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            log_file = self.model_name + '.txt'
            log_path = os.path.join(self.log_dir, log_file)
            if self.current_iter == 0:
                with open(log_path, 'w+') as logg:
                    logg.write(self.model_name)
                    logg.write('\n')
                    logg.write(text)
                    logg.write('\n')
            else:
                with open(log_path, 'a') as logg:
                    logg.write(text)
                    logg.write('\n')
                    logg.close()

    def train_step(self):

        epoch_loss = 0
        self.model.train()

        if self.train_dataloader is not None:

            for i, batch in enumerate(self.train_dataloader):

                input, label = self.data_extraction_func(batch)
                input = input.cuda()
                label = label.cuda().requires_grad_(False)

                self.optimizer.zero_grad()

                predicted_label = self.model(input)

                loss = self.loss_function(predicted_label, label)

                epoch_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                print('train epoch = {}, iter = {}/{}, loss = {}'.format(
                    self.current_iter, i, len(self.train_dataloader)-1, loss))

        return epoch_loss

    @torch.no_grad()
    def test_step(self):

        epoch_loss = 0
        self.model.eval()

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

        return epoch_loss