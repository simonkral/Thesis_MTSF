from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, _Simon_DWSC, _Linear_CD, _Linear_Delta_model, _Linear_Delta_regular, _Linear_final, _ModernTCN
from utils.tools import UnfreezeParam, EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric, metric_of_channels

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import wandb

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'Simon_DWSC': _Simon_DWSC,
            'Linear_CD': _Linear_CD,
            'Linear_Delta': _Linear_Delta_model,
            'Linear_Delta_reg': _Linear_Delta_regular,
            'Linear_final': _Linear_final,
            'ModernTCN': _ModernTCN
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.lambda_freeze_patience > 0:
            base_params = [p for name, p in self.model.named_parameters() if ('_cd_param' not in name and 'cd_regularization' not in name)]
            #model_optim = optim.Adam(base_params, lr=self.args.learning_rate)
            model_optim = optim.AdamW(base_params, lr=self.args.learning_rate)
        else:
            #model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            decay = []
            no_decay = []

            for name, param in self.model.named_parameters():
                if 'Linear_CD' in name:
                    decay.append(param)       # Apply weight decay
                    print(f"Applying weight decay to {name}")
                else:
                    no_decay.append(param)    # No weight decay
                    print(f"No weight decay for {name}")

            #model_optim = torch.optim.Adam([
            model_optim = torch.optim.AdamW([
                {'params': decay, 'weight_decay': self.args.cd_weight_decay},
                {'params': no_decay, 'weight_decay': 0.0}
            ], lr=self.args.learning_rate)

        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
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
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'Simon' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'Simon' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        unfreeze_param = UnfreezeParam(patience=self.args.lambda_freeze_patience)

        model_optim = self._select_optimizer()
        frozen_param = True
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'Simon' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'Simon' in self.args.model:
                            outputs = self.model(batch_x)
                    elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
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
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            unfreeze_param(vali_loss)
            if frozen_param and unfreeze_param.unfreeze and self.args.lambda_freeze_patience > 0:
                print("Unfreezing model parameters")
                frozen_param = False

                if hasattr(self.model, '_cd_param'):
                    self.model._cd_param.requires_grad = True
                    model_optim.add_param_group({'params': [self.model._cd_param]})

                if hasattr(self.model, 'cd_regularization'):
                    self.model._cd_param.cd_regularization = True
                    model_optim.add_param_group({'params': [self.model.cd_regularization]})

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

            # Log to wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": vali_loss,
                "test_loss": test_loss,
                "delta_model_lambda": (
                    self.model.cd_regularization.item()
                    if  hasattr(self.model, 'cd_regularization') and 
                        hasattr(self.model, 'sigmoid') and 
                        hasattr(self.model, 'learn_cd_regularization') and 
                        self.model.sigmoid == 0 and 
                        self.model.learn_cd_regularization == 1
                    else None
                ),
                "delta_model_lambda-sigmoid": (
                    torch.sigmoid(self.model._cd_param).item()
                    if  hasattr(self.model, '_cd_param') and 
                        hasattr(self.model, 'sigmoid') and 
                        hasattr(self.model, 'learn_cd_regularization') and 
                        self.model.sigmoid == 1 and 
                        self.model.learn_cd_regularization == 1
                    else None
                )
            })

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    #def test(self, setting, test=0):
    def test(self, setting, test=0, data_flag='test'):
        test_data, test_loader = self._get_data(flag=data_flag)
        
        #test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'Simon' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'Simon' in self.args.model:
                            outputs = self.model(batch_x)
                    elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)  # list of numpy arrays
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

        channel_metrics = metric_of_channels(preds, trues)

        mse_per_channel_list = [m['mse'] for m in channel_metrics]
        mae_per_channel_list = [m['mae'] for m in channel_metrics]
        
        # Create a table with indices and values
        mse_per_channel_table = wandb.Table(columns=["channel", "mse"])
        for i, val in enumerate(mse_per_channel_list):
            mse_per_channel_table.add_data(i, val)

        mae_per_channel_table = wandb.Table(columns=["channel", "mae"])
        for i, val in enumerate(mae_per_channel_list):
            mae_per_channel_table.add_data(i, val)

        if data_flag=='test':
            print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            f = open("result.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            f.write('\n')
            f.write('\n')
            f.close()

            # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            np.save(folder_path + 'x.npy', inputx)

            # Log to wandb:
            wandb.log({
                "mse": mse,
                "mae": mae,
                "rse": rse,
                #"mse_per_channel_table": mse_per_channel_table,
                "mse_per_channel_list": mse_per_channel_list,
                #"mae_per_channel_table": mae_per_channel_table,
                "mae_per_channel_list": mae_per_channel_list,
            })
        else:
            # Log to wandb:
            wandb.log({
                "mse_train": mse,
                "mae_train": mae,
                "rse_train": rse,
                #"mse_train_per_channel_table": mse_per_channel_table,
                "mse_train_per_channel_list": mse_per_channel_list,
                #"mae_train_per_channel_table": mae_per_channel_table,
                "mae_train_per_channel_list": mae_per_channel_list,
            })

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'Simon' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'Simon' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
