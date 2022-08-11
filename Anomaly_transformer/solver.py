from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from .utils.utils import *
from .model.AnomalyTransformer import AnomalyTransformer
from .data_factory.data_loader import get_loader_segment

from tqdm import tqdm

import streamlit as st


def my_kl_loss(p, q):
    # (bhls)
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1) # (bhls) -> (bhl) -> (bl) header의 평균


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.result_save_path, 'checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu" if torch.backends.cpu.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        now = datetime.now()
        self.time = now.strftime('%Y_%m_%d_%H_%M_%S')
        self.result_save_path = os.path.join("./result", f"{self.dataset}_{self.time}")

        os.mkdir(self.result_save_path)
        with open(os.path.join(self.result_save_path, "config.txt"),'w',encoding='UTF-8') as f:
            for code,name in config.items():
                f.write(f'{code} : {name}\n')

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=self.elayers, d_model=self.dmodel, d_ff=self.dff)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

        # if torch.backends.cpu.is_available():
        #     self.model.to(torch.device('cpu'))

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")

        rec_loss_list = []
        assdis_list = []

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in tqdm(enumerate(self.train_loader), desc="TRAINING"):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input) # (blhd, bhls, bhls? ,bhll) * num_layer

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (
                            torch.mean(
                                my_kl_loss(
                                    series[u], #(bhls)
                                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach()) #Rescale
                            ) + # torch.mean((bl))
                            torch.mean(
                                my_kl_loss(
                                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach(),
                                    series[u])
                            )   # torch.mean((bl))
                    ) # MAXIMIZE
                    prior_loss += (
                            torch.mean(
                                my_kl_loss(
                                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)),
                                    series[u].detach())
                            ) +
                            torch.mean(
                                my_kl_loss(
                                    series[u].detach(),
                                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)))
                            )
                    ) # MINIMIZE
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss # MINIMIZE
                loss2 = rec_loss + self.k * prior_loss  # MAXIMIZE

                rec_loss_list.append(rec_loss.detach().cpu().numpy())
                assdis_list.append(series_loss.detach().cpu().numpy())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                self.optimizer.step()
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))

            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

        np.save(os.path.join(self.result_save_path, f"{self.dataset}_rec_list"), np.array(rec_loss_list))
        np.save(os.path.join(self.result_save_path, f"{self.dataset}_assdis_list"), np.array(assdis_list))

    def test(self):
        criterion = nn.MSELoss(reduce=False)
        temperature = 50
        self.model.load_state_dict(
            torch.load(os.path.join(self.model_save_path, 'checkpoint.pth'), map_location=torch.device('cpu')))
        self.model.eval()

        try:
            combined_energy = np.load(os.path.join(self.model_save_path, "combined_energy.npy"))
            thresh = np.percentile(combined_energy, 100 - self.anomaly_ratio)
        except:
            self.model.load_state_dict(
                torch.load(os.path.join(self.model_save_path, 'checkpoint.pth'), map_location=torch.device('cpu')))
            self.model.eval()

            # (1) stastic on the train set
            with st.spinner(text="Stastic on the train set..."):
                my_bar = st.progress(0)
                attens_energy = []
                for i, (input_data, labels) in enumerate(self.train_loader):
                    my_bar.progress(int(i * (100 / len(self.train_loader))) + 1)
                    input = input_data.float().to(self.device)
                    output, series, prior, _ = self.model(input)
                    loss = torch.mean(criterion(input, output), dim=-1)
                    series_loss = 0.0
                    prior_loss = 0.0
                    for u in range(len(prior)):
                        if u == 0:
                            series_loss = my_kl_loss(series[u], (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach()) * temperature
                            prior_loss = my_kl_loss(
                                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size)),
                                series[u].detach()) * temperature
                        else:
                            series_loss += my_kl_loss(series[u], (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach()) * temperature
                            prior_loss += my_kl_loss(
                                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size)),
                                series[u].detach()) * temperature

                    metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                    cri = metric * loss
                    cri = cri.detach().cpu().numpy()
                    attens_energy.append(cri)
                my_bar.empty()

                attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
                train_energy = np.array(attens_energy)

            # (2) find the threshold
            with st.spinner(text="Finding the threshold..."):
                my_bar = st.progress(0)
                attens_energy = []
                for i, (input_data, labels) in enumerate(self.thre_loader):
                    my_bar.progress(int(i * (100 / len(self.thre_loader))) + 1)
                    input = input_data.float().to(self.device)
                    output, series, prior, _ = self.model(input)

                    loss = torch.mean(criterion(input, output), dim=-1)

                    series_loss = 0.0
                    prior_loss = 0.0
                    for u in range(len(prior)):
                        if u == 0:
                            series_loss = my_kl_loss(series[u], (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach()) * temperature
                            prior_loss = my_kl_loss(
                                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size)),
                                series[u].detach()) * temperature
                        else:
                            series_loss += my_kl_loss(series[u], (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach()) * temperature
                            prior_loss += my_kl_loss(
                                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size)),
                                series[u].detach()) * temperature
                    # Metric
                    metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                    cri = metric * loss
                    cri = cri.detach().cpu().numpy()
                    attens_energy.append(cri)
                my_bar.empty()

                attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
                test_energy = np.array(attens_energy)
                combined_energy = np.concatenate([train_energy, test_energy], axis=0)
                thresh = np.percentile(combined_energy, 100 - self.anomaly_ratio)
                print("Threshold :", thresh)

                np.save(os.path.join(self.model_save_path, "combined_energy"), test_energy)

        # (3) evaluation on the test set
        with st.spinner(text="Evaluation on the test set..."):
            my_bar = st.progress(0)
            test_labels = []
            attens_energy = []
            for i, (input_data, labels) in enumerate(self.thre_loader):
                my_bar.progress(int(i * (100 / len(self.thre_loader))) + 1)
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)

                loss = torch.mean(criterion(input, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)

                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels.append(labels)
            my_bar.empty()

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)
            test_labels = np.array(test_labels)

            np.save(os.path.join(self.model_save_path, f"test_energy"), test_energy)
            np.save(os.path.join(self.model_save_path, f"test_labels"), test_labels)

        # test_energy = np.load(os.path.join(self.model_save_path, "test_energy.npy"))
        # test_labels = np.load(os.path.join(self.model_save_path, "test_labels.npy"))
        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')

        # st.write(f"batch_size {self.batch_size}\nd_model {self.dmodel}\nd_ff {self.dff}\nk {self.k}\nwindow_size {self.win_size}")
        # st.write(f"e_layers {self.elayers}\npatience {self.patience}")
        # st.write(
        #     "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        #         accuracy, precision,
        #         recall, f_score))

        return accuracy, precision, recall, f_score, test_energy, thresh
