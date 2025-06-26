from data_provider.data_factory import data_provider_list as data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import vm_metric_with_mask as metric
import torch
import torch.nn as nn
from torch import optim, Tensor
from torch.utils.data import DataLoader
import os
import time
import warnings
import random
import numpy as np
from copy import deepcopy

warnings.filterwarnings('ignore')


class MaskedMSELoss(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        self.criterion = nn.MSELoss(size_average=size_average, reduce=reduce, reduction=reduction)
        self.keepdims = reduction != 'mean'

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mask = target != 0
        loss = self.criterion(input[mask], target[mask])

        if self.keepdims:
            masked_loss = torch.zeros_like(target)
            masked_loss[mask] = loss
            return masked_loss
        else:
            return loss


class Exp_Masked_Vital_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Masked_Vital_Forecast, self).__init__(args)

        if self.args.features == 'MP':
            self.f_dim = - abs(self.args.num_target)
        elif self.args.features == 'MS':
            self.f_dim = -1
        else:
            self.f_dim = 0

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

        num_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.model.parameters())
        print(
            f'Number of total parameters: {num_total_param / 1e6:.3f} Mb, tunable parameters: {num_param / 1e6:.3f} Mb')

        return model_optim

    def _select_criterion(self):
        reduction = 'none' if self.args.info_batch else 'mean'
        if self.args.loss in ['MSE', 'L2']:
            # criterion = nn.MSELoss(reduction=reduction)
            criterion = MaskedMSELoss(reduction=reduction)
        else:
            raise NotImplementedError

        if self.args.shuffle_mode == 5:
            criterion = MaskedMSELoss(reduction='none')

        return criterion

    def vali_(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, self.f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, self.f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                if self.args.shuffle_mode == 5:
                    loss = loss.mean()

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def vali_per_patient(self, vali_data_list, vali_loader_list, criterion):
        self.model.eval()

        total_loss_list = []
        for j, vali_loader in enumerate(vali_loader_list):
            preds = []
            trues = []
            with torch.no_grad():
                for i, (f_dim, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                    if self.args.shuffle_mode == 5 and self.args.masked_training:
                        self.model.set_reordering_index(vali_loader.indices)

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    self.f_dim = f_dim

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    # print(outputs.shape, batch_y.shape)
                    outputs = outputs[:, -self.args.pred_len:, self.f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, self.f_dim:].to(self.device)

                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

                    preds.append(pred)
                    trues.append(true)

            loss = criterion(torch.cat(preds, dim=0), torch.cat(trues, dim=0))

            if self.args.shuffle_mode == 5:
                loss = loss.mean()

            total_loss_list.append(loss.item())

            if self.args.shuffle_mode == 5 and self.args.masked_training:
                self.model.reset_ids_shuffle()

        total_loss = np.nanmean(total_loss_list)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        num_train = len(train_data)
        num_vali = len(vali_data)
        num_test = len(test_data)

        total_samples = num_train + num_vali + num_test

        print(f"Train samples: {num_train}")
        print(f"Validation samples: {num_vali}")
        print(f"Test samples: {num_test}")
        print(f"Total samples: {total_samples}")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        from utils.optim_utils import get_lr_scheduler
        total_steps = len(train_loader) * self.args.train_epochs
        if isinstance(self.args.warmup_steps, float):
            self.args.warmup_steps = int(self.args.warmup_steps * total_steps)
        # scheduler = get_lr_scheduler(model_optim, total_steps=total_steps, args=self.args)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (f_dim, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                if self.args.shuffle_mode == 5:
                    if self.args.masked_training:
                        self.model.set_reordering_index(train_loader.indices)

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                self.f_dim = f_dim

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        outputs = outputs[:, -self.args.pred_len:, self.f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, self.f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)

                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    outputs = outputs[:, -self.args.pred_len:, self.f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, self.f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)

                    if self.args.shuffle_mode == 5:
                        self.model.batch_update_state(loss)
                        loss = loss.mean()

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.nanmean(train_loss)
            # raise Exception
            vali_loss = self.vali_per_patient(vali_data, vali_loader, criterion)
            test_loss = self.vali_per_patient(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if not self.args.no_lradj:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data_list, test_loader_list = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        folder_path = f'{self.args.visualization}/{setting}/'
        os.makedirs(folder_path, exist_ok=True)

        mae_list, rmse_list, mape_list, r2_list, cc_list = [], [], [], [], []
        for j, (test_data, test_loader) in enumerate(zip(test_data_list, test_loader_list)):
            preds = []
            trues = []

            self.model.eval()
            with torch.no_grad():
                # patient by patient
                for i, (f_dim, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    if self.args.shuffle_mode == 5 and self.args.masked_training:
                        self.model.set_reordering_index(test_loader.indices)

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    if j < 6 and i < 1:
                        print(batch_x.shape)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    self.f_dim = f_dim

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    outputs = outputs[:, -self.args.pred_len:, :]

                    if torch.isnan(outputs).sum() > 0:
                        print(outputs)

                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                    outputs = outputs[:, :, self.f_dim:]
                    batch_y = batch_y[:, :, self.f_dim:]

                    pred = outputs
                    true = batch_y

                    preds.append(pred)
                    trues.append(true)

            preds = np.concatenate(preds, axis=0)  
            trues = np.concatenate(trues, axis=0)  

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

            # result save
            folder_path = f'{self.args.results}/{setting}/'
            os.makedirs(folder_path, exist_ok=True)

            mae, rmse, mape, r2, cc = metric(preds, trues)

            mae_list.append(mae)
            rmse_list.append(rmse)
            mape_list.append(mape)
            r2_list.append(r2)
            cc_list.append(cc)

            if self.args.shuffle_mode == 5 and self.args.masked_training:
                self.model.reset_ids_shuffle()

        msg = ""
        for score_list, score_name in zip([mae_list, rmse_list, mape_list, r2_list, cc_list],
                                          ['mae', 'rmse', 'mape', 'r2', 'cc']):
            try:
                score_mean = np.nanmean(score_list)
                score_std = np.nanstd(score_list)
                msg += f"{score_name}: {score_mean:.6f}±{score_std:.6f}, "
            except Exception as e:
                print(score_name)
                print(e)
        msg = msg.rstrip(', ')
        print(msg)

        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write(msg)
            f.write('\n\n')

        return

    def test_generalization(self, setting, test=0):
        assert self.args.domain_to_generalize is not None

        target_dict = {1: 'BIS/BIS', 2: 'MAP_ART'}
        num_target_dict = {1: 8, 2: 13}
        n_channel_dict = {1: 17, 2: 33}
        root_path_dict = {
            1: "./dataset/VitalDB/data/data_every_1m",
            2: "./dataset/MOVER-SIS",
        }

        root_path_to_bk = self.args.root_path
        target_to_bk = self.args.target
        num_target_to_bk = self.args.num_target
        enc_in_to_bk = dec_in_to_bk = c_out_to_bk = self.args.enc_in

        if self.args.domain_to_generalize == 2:
            self.args.data_split_json = "./dataset/VitalDB/MOVER-SIS.json"

        print(f"loading from {root_path_dict[self.args.domain_to_generalize]}...")
        self.args.root_path = root_path_dict[self.args.domain_to_generalize]
        self.args.target = target_dict[self.args.domain_to_generalize]
        self.args.num_target = num_target_dict[self.args.domain_to_generalize]
        self.args.enc_in = self.args.dec_in = self.args.c_out = n_channel_dict[self.args.domain_to_generalize]
        test_data_list, test_loader_list = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        folder_path = f'{self.args.visualization}/{setting}/'
        os.makedirs(folder_path, exist_ok=True)

        mae_list, rmse_list, mape_list, r2_list, cc_list = [], [], [], [], []
        for j, (test_data, test_loader) in enumerate(zip(test_data_list, test_loader_list)):
            preds = []
            trues = []

            self.model.eval()
            with torch.no_grad():
                # patient by patient
                for i, (f_dim, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    if self.args.shuffle_mode == 5 and self.args.masked_training:
                        self.model.set_reordering_index(test_loader.indices)

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    if j < 6 and i < 1:
                        print(batch_x.shape)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    self.f_dim = f_dim

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    outputs = outputs[:, -self.args.pred_len:, :]

                    if torch.isnan(outputs).sum() > 0:
                        print(outputs)

                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                    outputs = outputs[:, :, self.f_dim:]
                    batch_y = batch_y[:, :, self.f_dim:]

                    pred = outputs
                    true = batch_y

                    preds.append(pred)
                    trues.append(true)

            preds = np.concatenate(preds, axis=0)  # e.g. [(2048, 360, 9), (2048, 360, 9), (706, 360, 9)]
            trues = np.concatenate(trues, axis=0)  # e.g. [(2048, 360, 9), (2048, 360, 9), (706, 360, 9)]

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

            # result save
            folder_path = f'{self.args.results}/{setting}/'
            os.makedirs(folder_path, exist_ok=True)

            mae, rmse, mape, r2, cc = metric(preds, trues)

            mae_list.append(mae)
            rmse_list.append(rmse)
            mape_list.append(mape)
            r2_list.append(r2)
            cc_list.append(cc)

            if self.args.shuffle_mode == 5 and self.args.masked_training:
                self.model.reset_ids_shuffle()

        msg = ""
        for score_list, score_name in zip([mae_list, rmse_list, mape_list, r2_list, cc_list],
                                          ['mae', 'rmse', 'mape', 'r2', 'cc']):
            try:
                score_mean = np.nanmean(score_list)
                score_std = np.nanstd(score_list)
                msg += f"{score_name}: {score_mean:.6f}±{score_std:.6f}, "
            except Exception as e:
                print(score_name)
                print(e)
        msg = msg.rstrip(', ')
        print(msg)

        with open("result_long_term_forecast_dg.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write(msg)
            f.write('\n\n')

        # recover
        self.args.root_path = root_path_to_bk
        self.args.target = target_to_bk
        self.args.num_target = num_target_to_bk
        self.args.enc_in = self.args.dec_in = self.args.c_out = enc_in_to_bk

        if self.args.domain_to_generalize == 2:
            self.args.data_split_json = None

        return
