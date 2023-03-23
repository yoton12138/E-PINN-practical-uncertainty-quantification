import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torchbnn as bnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from options import Opt
from sklearn.metrics import r2_score
from utils import *


class Fnn(nn.Module):
    def __init__(self, layers_size, st=False, dropout=False):
        super().__init__()

        self.st = st
        self.model = nn.Sequential()
        self.activation_func = nn.Tanh()
        # 参数化网络结构
        if len(layers_size) < 4:
            raise ValueError("网络结构太浅啦")
        i_layer_size = layers_size[0]
        o_layer_size = layers_size[-1]
        h_layer_list = layers_size[1:-1]

        self.input = nn.Linear(i_layer_size, h_layer_list[0])
        self.model.add_module("input_layer", self.input)
        self.model.add_module("ac_fun", self.activation_func)

        for i in range(len(h_layer_list)-1):
            layer_name = "hidden_layer_" + str(i)
            self.model.add_module(layer_name, nn.Linear(h_layer_list[i], h_layer_list[i + 1]))
            self.model.add_module("ac_fun"+str(i), self.activation_func)
            if dropout:
                self.model.add_module("dropout_" + str(i), nn.Dropout(p=0.1))

        self.output = nn.Linear(h_layer_list[-1], o_layer_size)
        self.model.add_module("output_layer", self.output)

    def forward(self, x):
        out = self.model(x)
        if self.st:
            out = F.softplus(out) + 1e-6
        return out


class BNNNet(torch.nn.Module):
    def __init__(self, layers_size, activation_function=torch.nn.Tanh,
                 prior_weight_mu=0., prior_weight_sigma=0.2, st=False):
        super().__init__()

        self.st = st
        self.layers = layers_size
        if len(layers_size) < 3:
            raise ValueError("网络结构太浅")

        self.model = torch.nn.Sequential()
        for i in range(len(self.layers) - 2):
            layer = bnn.BayesLinear(prior_weight_mu, prior_weight_sigma, self.layers[i], self.layers[i+1])
            act = activation_function()
            self.model.add_module(str(i), layer)
            self.model.add_module(str(i) + "_act", act)

        self.output = bnn.BayesLinear(prior_weight_mu, prior_weight_sigma, self.layers[-2], self.layers[-1])
        self.model.add_module("out", self.output)

    def forward(self, x):
        out = self.model(x)
        if self.st:
            out = F.softplus(out) + 1e-6

        return out


class PINN(object):
    def __init__(self, layers_size, in_obs, out_obs, collocation_points, source,
                 in_test=None, out_test=None, x_bc=None, y_bc=None):
        self.net_u = Fnn(layers_size).to(Opt.device)
        self.in_obs = in_obs
        self.out_obs = out_obs
        self.collocation_points = collocation_points
        self.X_f = collocation_points[:-2000, :]
        self.X_star = collocation_points[-2000:, :]
        self.net_s = source
        self.in_test = in_test
        self.out_test = out_test
        self.x_bc = x_bc
        self.y_bc = y_bc

        self.mse_loss = torch.nn.MSELoss()
        self.nll_loss = torch.nn.GaussianNLLLoss()

        self.lw_mse = Opt.lw_mse
        self.lw_pde = Opt.lw_pde

        self.lr_u = Opt.lr_u
        self.lr_s = Opt.lr_s

        self.optimizer = torch.optim.Adam([
                                           {"params": self.net_u.parameters(), "lr": self.lr_u},
                                           {"params": self.net_s.parameters(), "lr": self.lr_s}
                                          ], betas=(0.9, 0.999), eps=1e-08)
        self.epoch = 0
        self.M = Opt.M
        self.loss_list = []
        self.loss_m = []
        self.model_param_list = []

    def pde_loss(self, inputs):
        _, _, _, du_xx, du_yy = Gradients(inputs, self.net_u).second_order()
        s = self.net_s(inputs).view(-1, 1)
        res = 0.02 * (du_xx + du_yy) + s
        return torch.mean(res**2)

    def bc_loss(self, x_bc, y_bc):
        _, du_x, _ = Gradients(x_bc, self.net_u).first_order()
        _, _, du_y = Gradients(y_bc, self.net_u).first_order()
        return torch.mean(du_x**2) + torch.mean(du_y)**2

    def loss(self):
        out_pred = self.net_u(self.in_obs)
        mse = self.mse_loss(out_pred, self.out_obs)
        pde = self.pde_loss(self.collocation_points)

        # adversarial training
        # in_obs_at = self.fgm(mse, self.in_obs)
        # out_pred_at = self.net_u(in_obs_at)
        # mse_ = self.mse_loss(out_pred_at, self.out_obs)
        # mse = mse + mse_
        #
        loss = self.lw_mse*mse + self.lw_pde*pde
        self.optimizer.zero_grad()
        loss.backward()
        self.epoch += 1
        self.loss_m.append(loss.item())

        if self.epoch % 1000 == 0:
            print('epoch %d, Loss: %.4e, Loss_mse: %.4e, Loss_pde: %.4e'
                  % (self.epoch, loss.item(), self.lw_mse*mse.item(), self.lw_pde*pde.item()))

        if self.epoch % 5000 == 0:
            s_pred_test = self.net_s(self.in_test)
            s_pred_test = s_pred_test.detach().cpu().numpy()
            r2 = r2_score(self.out_test, s_pred_test)
            l2_error = np.linalg.norm(self.out_test.reshape(-1, 1) - s_pred_test, 2) / np.linalg.norm(self.out_test, 2)
            print("sample r2: ", r2, " l2_error: ", l2_error)

        return loss

    def train(self, epochs=10000):
        for m in range(self.M):
            print("-----训练开始-----")
            for i in range(self.epoch, epochs + 1):
                self.optimizer.step(self.loss)

            stat_dict = self.net_s.state_dict()
            self.model_param_list.append(stat_dict)
            self.loss_list.append(self.loss_m)

            # 重新初始化一个模型
            self.net_u = Fnn(layers_size=Opt.layers_u).to(Opt.device)
            self.net_s = Fnn(layers_size=Opt.layers_s, st=True).to(Opt.device)
            self.optimizer = torch.optim.Adam([{"params": self.net_u.parameters(), "lr": self.lr_u},
                                               {"params": self.net_s.parameters(), "lr": self.lr_s}
                                               ], betas=(0.9, 0.999), eps=1e-08)

            self.epoch = 0
            self.loss_m = []

    def train_batch(self, epochs=1000):
        in_obs_d = Dataset(self.in_obs)
        out_obs_d = Dataset(self.out_obs)
        batch_size_obs = 256
        in_obs_train = torch.utils.data.DataLoader(in_obs_d, batch_size=batch_size_obs)
        out_obs_train = torch.utils.data.DataLoader(out_obs_d, batch_size=batch_size_obs)

        for m in range(self.M):
            for epoch in range(epochs + 1):
                for in_obs_b, out_obs_b in zip(in_obs_train, out_obs_train):
                    out_pred = self.net_u(in_obs_b)
                    mse = self.mse_loss(out_pred, out_obs_b)
                    pde = self.pde_loss(self.collocation_points)

                    # adversarial training
                    in_obs_at = self.fgm(mse, in_obs_b)
                    out_pred_at = self.net_u(in_obs_at)
                    mse_ = self.mse_loss(out_pred_at, out_obs_b)
                    mse = mse + mse_

                    loss = self.lw_mse * mse + self.lw_pde * pde

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.loss_m.append(loss.item())
                if epoch % 200 == 0:
                    print('epoch %d, Loss: %.4e, Loss_mse: %.4e, Loss_pde: %.4e'
                          % (epoch, loss.item(), self.lw_mse * mse.item(), self.lw_pde * pde.item()))

                if epoch % 500 == 0:
                    s_pred_test = self.net_s(self.in_test)
                    s_pred_test = s_pred_test.detach().cpu().numpy()
                    r2 = r2_score(self.out_test, s_pred_test)
                    l2_error = np.linalg.norm(self.out_test - s_pred_test, 2) / np.linalg.norm(self.out_test, 2)
                    print("sample r2: ", r2, " l2_error: ", l2_error)

            stat_dict = self.net_s.state_dict()
            self.model_param_list.append(stat_dict)
            self.loss_list.append(self.loss_m)
            # 重新初始化一个模型
            self.net_u = Fnn(layers_size=Opt.layers_u, st=True).to(Opt.device)
            self.net_s = Fnn(layers_size=Opt.layers_s, st=True).to(Opt.device)
            self.optimizer = torch.optim.Adam([{"params": self.net_u.parameters(), "lr": self.lr_u},
                                               {"params": self.net_s.parameters(), "lr": self.lr_s}
                                               ], betas=(0.9, 0.999), eps=1e-08)
            self.epoch = 0
            self.loss_m = []

    def predict(self, x_test):
        """利用M个模型加权输出
        """
        model = Fnn(layers_size=Opt.layers_s, st=True).to(Opt.device)
        pre_list = []
        for param in self.model_param_list:
            model.load_state_dict(param)
            v_pre = model(x_test).data.cpu().numpy()
            pre_list.append(v_pre)
        mean = np.mean(pre_list, axis=0)
        std = np.std(pre_list, axis=0)

        return mean, std

    def predict_m(self, x_test, m=0):
        """某个个模型输出
        """
        model = Fnn(layers_size=Opt.layers_s, st=True).to(Opt.device)
        param = self.model_param_list[m]
        model.load_state_dict(param)
        s_pre = model(x_test).data.cpu().numpy()

        return s_pre

    def fgm(self, l, inputs, eps=0.01):
        dl_X = torch.autograd.grad(l, inputs, torch.ones_like(l), retain_graph=True, create_graph=True)[0]
        norms = torch.norm(dl_X, 2, dim=1).view(-1, 1)  # 这种方式快,也需要把0和nan给替换成大数,保持摄动极小
        norms = torch.where(torch.isnan(norms), torch.full_like(norms, 1e16), norms)
        norms = torch.where(norms == 0, torch.full_like(norms, 1e16), norms)
        r_at = eps * dl_X/norms
        X_at = inputs + r_at

        return X_at


class Gradients(object):
    def __init__(self, inputs, net_u):
        self.inputs = inputs
        self.outs = net_u(inputs)
        self.net_u = net_u

    def first_order(self):
        """输出有必要的一阶导数"""
        du_X = torch.autograd.grad(self.outs, self.inputs, torch.ones_like(self.outs),
                                   retain_graph=True, create_graph=True)[0]
        du_x = du_X[:, 0].view(-1, 1)
        du_y = du_X[:, 1].view(-1, 1)

        return self.outs, du_x, du_y

    def second_order(self):
        """输出有必要的二阶导数"""
        _, du_x, du_y = self.first_order()
        du_xX = torch.autograd.grad(du_x, self.inputs, torch.ones_like(du_x),
                                    retain_graph=True, create_graph=True)[0]
        du_xx = du_xX[:, 0].view(-1, 1)

        du_yX = torch.autograd.grad(du_y, self.inputs, torch.ones_like(du_y),
                                    retain_graph=True, create_graph=True)[0]
        du_yy = du_yX[:, 1].view(-1, 1)

        return self.outs, du_x, du_y, du_xx, du_yy


class MCDropout(object):
    def __init__(self, layers_size, in_obs, out_obs, collocation_points, source,
                 in_test=None, out_test=None, x_bc=None, y_bc=None):
        self.net_u = Fnn(layers_size).to(Opt.device)
        self.in_obs = in_obs
        self.out_obs = out_obs
        self.collocation_points = collocation_points
        self.net_s = source
        self.in_test = in_test
        self.out_test = out_test
        self.x_bc = x_bc
        self.y_bc = y_bc

        self.mse_loss = torch.nn.MSELoss()
        self.nll_loss = torch.nn.GaussianNLLLoss()

        self.lw_mse = Opt.lw_mse
        self.lw_pde = Opt.lw_pde

        self.lr_u = Opt.lr_u
        self.lr_s = Opt.lr_s

        self.optimizer = torch.optim.Adam([
                                           {"params": self.net_u.parameters(), "lr": self.lr_u},
                                           {"params": self.net_s.parameters(), "lr": self.lr_s}
                                          ], betas=(0.9, 0.999), eps=1e-08)

        self.epoch = 0
        self.loss_list = []
        self.s_stat_dict = None

    def pde_loss(self, inputs):
        _, _, _, du_xx, du_yy = Gradients(inputs, self.net_u).second_order()
        s = self.net_s(inputs).view(-1, 1)
        res = 0.02 * (du_xx + du_yy) + s
        return torch.mean(res**2)

    def bc_loss(self, x_bc, y_bc):
        _, du_x, _ = Gradients(x_bc, self.net_u).first_order()
        _, _, du_y = Gradients(y_bc, self.net_u).first_order()
        return torch.mean(du_x**2) + torch.mean(du_y)**2

    def loss(self):
        out_pred = self.net_u(self.in_obs)
        mse = self.mse_loss(out_pred, self.out_obs)
        pde = self.pde_loss(self.collocation_points)
        #
        loss = self.lw_mse*mse + self.lw_pde*pde
        self.optimizer.zero_grad()
        loss.backward()
        self.epoch += 1
        self.loss_list.append(loss.item())

        if self.epoch % 1000 == 0:
            print('epoch %d, Loss: %.4e, Loss_mse: %.4e, Loss_pde: %.4e'
                  % (self.epoch, loss.item(), self.lw_mse*mse.item(), self.lw_pde*pde.item()))

        if self.epoch % 5000 == 0:
            s_pred_test = self.net_s(self.in_test)
            s_pred_test = s_pred_test.detach().cpu().numpy()
            r2 = r2_score(self.out_test, s_pred_test)
            l2_error = np.linalg.norm(self.out_test.reshape(-1, 1) - s_pred_test, 2) / np.linalg.norm(self.out_test, 2)
            print("sample r2: ", r2, " l2_error: ", l2_error)

        return loss

    def train(self, epochs=10000):
        for i in range(0, epochs + 1):
            self.optimizer.step(self.loss)

        self.s_stat_dict = self.net_s.state_dict()

    def predict(self, x_test, M=100):
        """利用M个模型加权输出
        """
        pre_list = []
        for i in range(M):
            s_pre = self.net_s(x_test)
            s_pre_np = s_pre.data.cpu().numpy()
            pre_list.append(s_pre_np)

        return pre_list


class BPINN(object):
    def __init__(self, layers_size, in_obs, out_obs, collocation_points, source,
                 in_test=None, out_test=None, x_bc=None, y_bc=None):
        self.net_u = Fnn(layers_size).to(Opt.device)
        self.in_obs = in_obs
        self.out_obs = out_obs
        self.collocation_points = collocation_points
        self.X_f = collocation_points[:-2000, :]
        self.X_star = collocation_points[-2000:, :]
        self.net_s = source
        self.in_test = in_test
        self.out_test = out_test
        self.x_bc = x_bc
        self.y_bc = y_bc
        self.zeros = torch.zeros([collocation_points.shape[0], 1]).to(Opt.device)
        self.var = torch.zeros([collocation_points.shape[0], 1]).to(Opt.device) + 0.02**2

        self.mse_loss = torch.nn.MSELoss()
        self.nll_loss = torch.nn.GaussianNLLLoss()
        self.kl_loss = bnn.BKLLoss()

        self.lw_mse = Opt.lw_mse
        self.lw_pde = Opt.lw_pde
        self.lw_kl = Opt.lw_kl

        self.lr_u = Opt.lr_u
        self.lr_s = Opt.lr_s

        self.optimizer = torch.optim.Adam([{"params": self.net_u.parameters(), "lr": self.lr_u},
                                           {"params": self.net_s.parameters(), "lr": self.lr_s}
                                          ], betas=(0.9, 0.999), eps=1e-08)

        self.epoch = 0
        self.loss_list = []
        self.s_stat_dict = None

    def pde_loss(self, inputs):
        _, _, _, du_xx, du_yy = Gradients(inputs, self.net_u).second_order()
        s = self.net_s(inputs).view(-1, 1)
        res = 0.02 * (du_xx + du_yy) + s
        loss = torch.mean(res**2)  # MSE

        return loss

    def bc_loss(self, x_bc, y_bc):
        _, du_x, _ = Gradients(x_bc, self.net_u).first_order()
        _, _, du_y = Gradients(y_bc, self.net_u).first_order()
        return torch.mean(du_x**2) + torch.mean(du_y)**2

    def loss(self):
        out_pred = self.net_u(self.in_obs)
        mse = self.mse_loss(out_pred, self.out_obs)
        pde = self.pde_loss(self.collocation_points)
        kl = self.kl_loss(self.net_s)

        #
        loss = self.lw_mse*mse + self.lw_pde*pde + self.lw_kl*kl
        self.optimizer.zero_grad()
        loss.backward()
        self.epoch += 1
        self.loss_list.append(loss.item())

        if self.epoch % 1000 == 0:
            print('epoch %d, Loss: %.4e, Loss_mse: %.4e, Loss_pde: %.4e, Loss_kl: %.4e'
                  % (self.epoch, loss.item(), self.lw_mse*mse.item(), self.lw_pde*pde.item(), self.lw_kl*kl.item()))

        if self.epoch % 5000 == 0:
            u_r2 = r2_score(out_pred.data.cpu().numpy(), self.out_obs.data.cpu().numpy())
            print("temperature u_r2: ", u_r2)
            s_pred_test = self.net_s(self.in_test)
            s_pred_test = s_pred_test.detach().cpu().numpy()
            r2 = r2_score(self.out_test, s_pred_test)
            l2_error = np.linalg.norm(self.out_test.reshape(-1, 1) - s_pred_test, 2) / np.linalg.norm(self.out_test, 2)
            print("one sample r2: ", r2, " l2_error: ", l2_error)

            nx = 50
            ny = 50
            xx = np.linspace(0, 1, nx)
            yy = np.linspace(0, 1, ny)
            xxx, yyy = np.meshgrid(xx, yy)
            s_pred_test = s_pred_test.reshape(50, 50)
            fig = plt.figure(figsize=[4, 3])
            plt.contourf(xxx, yyy, s_pred_test, cmap="hot", levels=101)
            plt.colorbar()
            plt.savefig(f"training_results/iter_{self.epoch}.png")
            plt.close()

        return loss

    def train(self, epochs=10000):
        print("-----训练开始-----")
        for i in range(self.epoch, epochs + 1):
            self.optimizer.step(self.loss)

        self.s_stat_dict = self.net_s.state_dict()

    def predict(self, x_test, M=100):
        """利用M个模型加权输出
        """
        pre_list = []
        for i in range(M):
            s_pre = self.net_s(x_test)
            s_pre_np = s_pre.data.cpu().numpy()
            pre_list.append(s_pre_np)

        return pre_list
