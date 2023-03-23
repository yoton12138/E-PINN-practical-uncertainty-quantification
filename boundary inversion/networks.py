import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from options import Opt
from sklearn.metrics import r2_score
from utils import *
import torchbnn as bnn


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
            st = (x[:, 1:2] - 0.5)
            out = out * st  # 硬约束，当y=0.5时 输出为0满足一类边界
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
            st = (x[:, 1:2] - 0.5)
            out = out * st  # 硬约束，当y=0.5时 输出为0满足一类边界

        return out


class PINN(object):
    def __init__(self, layers_size, in_obs, out_obs, collocation_points, bc_left_right,
                 in_test=None, out_test=None):
        self.net_u = Fnn(layers_size, st=True).to(Opt.device)
        self.in_obs = in_obs
        self.out_obs = out_obs
        self.collocation_points = collocation_points
        self.xt_bc_left_right = bc_left_right
        self.in_test = in_test
        self.out_test = out_test

        self.mse_loss = torch.nn.MSELoss()
        self.nll_loss = torch.nn.GaussianNLLLoss()

        self.lw_mse = 1.

        self.lw_pde = Opt.lw_pde

        self.lr = Opt.lr

        self.optimizer = torch.optim.Adam([{"params": self.net_u.parameters(), "lr": self.lr}
                                           ], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)

        self.epoch = 0
        self.M = Opt.M
        self.loss_list = []
        self.loss_m = []
        self.model_param_list = []

    def pde_loss(self):
        u, du_x, du_y, du_t, du_xx, du_yy = Gradients(self.collocation_points, self.net_u).second_order()
        res = du_t - (du_xx + du_yy)*(1 + torch.exp(-u))  # 非线性的热传导率 ρc=1  k=1+e^-u

        return torch.mean(res**2)

    def bc_loss_left_right(self):
        """左右约束一下"""
        u, _, du_y, _ = Gradients(self.xt_bc_left_right, self.net_u).first_order()
        q = - (1 + torch.exp(-u)) * du_y

        return torch.mean(q**2)

    def predict_bc(self, xt_bc):
        """inductive net for predicting bc"""
        u, _, du_y, _ = Gradients(xt_bc, self.net_u).first_order()
        q = - (1 + torch.exp(-u)) * du_y

        return q

    def loss(self):
        out_pred = self.net_u(self.in_obs)
        mse = self.mse_loss(out_pred, self.out_obs)

        # adversarial training
        in_obs_at = self.fgm(mse, self.in_obs)
        out_pred_at = self.net_u(in_obs_at)
        mse_ = self.mse_loss(out_pred_at, self.out_obs)
        mse = mse + mse_

        pde = self.pde_loss()
        bc = self.bc_loss_left_right()

        loss = self.lw_mse*mse + self.lw_pde*pde + bc

        self.optimizer.zero_grad()
        loss.backward()
        self.epoch += 1
        self.loss_m.append(loss.item())

        if self.epoch % 1000 == 0:
            print('epoch %d, Loss: %.4e, Loss_mse: %.4e, Loss_pde: %.4e'
                  % (self.epoch, loss.item(), self.lw_mse*mse.item(), self.lw_pde*pde.item()))

        if self.epoch % 5000 == 0:
            bc_pred_test = self.predict_bc(self.in_test)
            bc_pred_test = bc_pred_test.detach().cpu().numpy()
            r2 = r2_score(self.out_test, np.squeeze(bc_pred_test))
            l2_error = np.linalg.norm(self.out_test - np.squeeze(bc_pred_test), 2) \
                       / np.linalg.norm(self.out_test, 2)
            print("bc r2: ", r2, " l2_error: ", l2_error)

            fig = plt.figure(figsize=[4, 3])
            in_test_np = self.in_test.detach().cpu().numpy()[:, 0]
            plt.plot(in_test_np, self.out_test)
            plt.plot(in_test_np, bc_pred_test, "--")
            plt.savefig(f"training_results/iter_{self.epoch}.png")
            plt.close()

            out_pred_np = out_pred.detach().cpu().numpy()
            x_lin, y_lin = np.linspace(-1, 1, 21), np.linspace(-0.5, 0.5, 11)
            x_mesh, y_mesh = np.meshgrid(x_lin, y_lin)

            fig = plt.figure(figsize=[4, 3])
            plt.contourf(x_mesh, y_mesh, out_pred_np[-231:].reshape(11, 21), cmap="jet", levels=11)
            plt.colorbar()
            plt.savefig(f"training_results/u_iter_{self.epoch}.png")
            plt.close()

        return loss

    def train(self, epochs=10000):
        for m in range(self.M):
            start_time = time.time()
            for i in range(self.epoch, epochs + 1):
                self.optimizer.step(self.loss)

            end_time = time.time()
            print(f"单个模型训练时间： {end_time - start_time}")

            stat_dict = self.net_u.state_dict()
            self.model_param_list.append(stat_dict)
            path = "save_models/params_"
            torch.save(stat_dict, path + str(m))
            self.loss_list.append(self.loss_m)
            # 重新初始化一个模型
            self.net_u = Fnn(layers_size=Opt.layers_u, st=True).to(Opt.device)
            self.optimizer = torch.optim.Adam([{"params": self.net_u.parameters(), "lr": self.lr},
                                               ], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)

            self.epoch = 0
            self.loss_m = []

    def predict(self, x_test):
        """利用M个模型加权输出QoI
        """
        self.net_u = Fnn(layers_size=Opt.layers_u, st=True).to(Opt.device)
        pred_list = []
        for param in self.model_param_list:
            self.net_u.load_state_dict(param)
            pred = self.predict_bc(x_test).data.cpu().numpy()
            pred_list.append(pred)
        mean = np.mean(pred_list[1:], axis=0)
        std = np.std(pred_list[1:], axis=0)

        return mean, std

    def predict_m(self, x_test, m=0):
        """某个个模型输出
        """
        self.net_u = Fnn(layers_size=Opt.layers_u, st=True).to(Opt.device)
        param = self.model_param_list[m]
        self.net_u.load_state_dict(param)
        pred = self.net_u.predict_bc(x_test).data.cpu().numpy()

        return pred

    def fgm(self, l, inputs, eps=0.01):
        dl_X = torch.autograd.grad(l, inputs, torch.ones_like(l), retain_graph=True, create_graph=True)[0]
        norms = torch.norm(dl_X, 2, dim=1).view(-1, 1)  # 这种方式快,也需要把0和nan给替换成大数,保持摄动极小
        norms = torch.where(torch.isnan(norms), torch.full_like(norms, 1e16), norms)
        norms = torch.where(norms == 0, torch.full_like(norms, 1e16), norms)
        r_at = eps*dl_X/norms
        X_at = inputs + r_at

        return X_at

    def save_models(self, m):
        if not os.path.exists('save_models/'):
            os.makedirs('save_models/')
        stat_dict = self.net_u.state_dict()
        path = "save_models/params_" + str(m) + ".pt"
        torch.save(stat_dict, path)

    def load_models(self, path_list):
        for path in path_list:
            stat_dict = torch.load(path)
            self.model_param_list.append(stat_dict)


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
        du_t = du_X[:, 2].view(-1, 1)

        return self.outs, du_x, du_y, du_t

    def second_order(self):
        """输出有必要的二阶导数"""
        _, du_x, du_y, du_t = self.first_order()

        du_xX = torch.autograd.grad(du_x, self.inputs, torch.ones_like(du_x),
                                    retain_graph=True, create_graph=True)[0]
        du_xx = du_xX[:, 0].view(-1, 1)

        du_yX = torch.autograd.grad(du_y, self.inputs, torch.ones_like(du_y),
                                    retain_graph=True, create_graph=True)[0]
        du_yy = du_yX[:, 1].view(-1, 1)

        return self.outs, du_x, du_y, du_t, du_xx, du_yy


class BPINN(object):
    def __init__(self, layers_size, in_obs, out_obs, collocation_points, bc_left_right,
                 in_test=None, out_test=None):
        self.net_u = BNNNet(layers_size, st=True).to(Opt.device)
        self.in_obs = in_obs
        self.out_obs = out_obs
        self.collocation_points = collocation_points
        self.xt_bc_left_right = bc_left_right
        self.in_test = in_test
        self.out_test = out_test

        self.var = torch.zeros_like(out_obs, dtype=torch.float32, requires_grad=False) + Opt.noise**2

        self.mse_loss = torch.nn.MSELoss()
        self.nll_loss = torch.nn.GaussianNLLLoss()
        self.kl_loss = bnn.BKLLoss(reduction="mean")

        self.lw_mse = 1.
        self.lw_pde = Opt.lw_pde
        self.lw_kl = Opt.lw_kl
        self.lw_bc = Opt.lw_bc

        self.lr = Opt.lr

        self.optimizer = torch.optim.Adam([{"params": self.net_u.parameters(), "lr": self.lr}],
                                          betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)

        self.epoch = 0
        self.loss_list = []
        self.loss_pde = []
        self.loss_bc = []

    def pde_loss(self):
        u, du_x, du_y, du_t, du_xx, du_yy = Gradients(self.collocation_points, self.net_u).second_order()
        res = du_t - (du_xx + du_yy)*(1 + torch.exp(-u))  # 非线性的热传导率 ρc=1  k=1+e^-u

        return torch.mean(res**2)

    def bc_loss_left_right(self):
        """左右约束一下"""
        u, _, du_y, _ = Gradients(self.xt_bc_left_right, self.net_u).first_order()
        q = - (1 + torch.exp(-u)) * du_y

        return torch.mean(q**2)

    def predict_bc(self, xt_bc):
        """inductive net for predicting bc"""
        u, _, du_y, _ = Gradients(xt_bc, self.net_u).first_order()
        q = - (1 + torch.exp(-u)) * du_y

        return q

    def loss(self):
        out_pred = self.net_u(self.in_obs)
        mse = self.nll_loss(out_pred, self.out_obs, self.var)  # 使用MSE的情况之下需要仔细调整权值，或者没有观测方差项
        kl = self.kl_loss(self.net_u)

        pde = self.pde_loss()
        bc = self.bc_loss_left_right()

        loss = self.lw_mse*mse + self.lw_kl*kl + self.lw_pde*pde + self.lw_bc*bc

        self.optimizer.zero_grad()
        loss.backward()
        self.epoch += 1
        self.loss_list.append(loss.detach().cpu().numpy())
        self.loss_pde.append(pde.detach().cpu().numpy())
        self.loss_bc.append(bc.detach().cpu().numpy())

        if self.epoch % 1000 == 0:
            print('epoch %d, Loss: %.4e, Loss_mse: %.4e, Loss_kl: %.4e, Loss_pde: %.4e'
                  % (self.epoch, loss.item(), self.lw_mse*mse.item(), self.lw_kl*kl.item(), self.lw_pde*pde.item()))

        if self.epoch % 5000 == 0:
            bc_pred_test = self.predict_bc(self.in_test)
            bc_pred_test = bc_pred_test.detach().cpu().numpy()
            r2 = r2_score(self.out_test, np.squeeze(bc_pred_test))
            l2_error = np.linalg.norm(self.out_test - np.squeeze(bc_pred_test), 2) \
                       / np.linalg.norm(self.out_test, 2)
            print("one sample bc r2: ", r2, " l2_error: ", l2_error)

            fig = plt.figure(figsize=[4, 3])
            in_test_np = self.in_test.detach().cpu().numpy()[:, 0]
            plt.plot(in_test_np, self.out_test)
            plt.plot(in_test_np, bc_pred_test, "--")
            plt.savefig(f"training_results/iter_{self.epoch}.png")
            plt.close()

            out_pred_np = out_pred.detach().cpu().numpy()
            x_lin, y_lin = np.linspace(-1, 1, 21), np.linspace(-0.5, 0.5, 11)
            x_mesh, y_mesh = np.meshgrid(x_lin, y_lin)

            fig = plt.figure(figsize=[4, 3])
            plt.contourf(x_mesh, y_mesh, out_pred_np[-231:].reshape(11, 21), cmap="jet", levels=11)
            plt.colorbar()
            plt.savefig(f"training_results/u_iter_{self.epoch}.png")
            plt.close()
            print(f"one sample u_r2: {r2_score(self.out_obs.data.cpu().numpy(), out_pred_np)}")

        return loss

    def train(self, epochs):
        for i in range(self.epoch, epochs + 1):
            self.optimizer.step(self.loss)

    def predict_u(self, inputs):
        predict_out = self.net_u(inputs)
        predict_out = predict_out.detach().cpu().numpy()

        return predict_out


class MCDropout(PINN):
    def __init__(self, layers_size, in_obs, out_obs, collocation_points, bc_left_right,
                 in_test=None, out_test=None):

        self.net_u = Fnn(layers_size, st=True, dropout=True).to(Opt.device)
        self.in_obs = in_obs
        self.out_obs = out_obs
        self.collocation_points = collocation_points
        self.xt_bc_left_right = bc_left_right
        self.in_test = in_test
        self.out_test = out_test

        self.mse_loss = torch.nn.MSELoss()

        self.lw_mse = 1.

        self.lw_pde = Opt.lw_pde

        self.lr = Opt.lr

        self.optimizer = torch.optim.Adam([{"params": self.net_u.parameters(), "lr": self.lr}
                                           ], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)

        self.epoch = 0
        self.M = Opt.M
        self.loss_list = []
        self.loss_pde = []
        self.loss_bc = []

    def loss(self):
        out_pred = self.net_u(self.in_obs)
        mse = self.mse_loss(out_pred, self.out_obs)

        pde = self.pde_loss()
        bc = self.bc_loss_left_right()

        loss = self.lw_mse * mse + self.lw_pde * pde + bc

        self.optimizer.zero_grad()
        loss.backward()
        self.epoch += 1
        self.loss_list.append(loss.item())
        self.loss_pde.append(pde.item())
        self.loss_bc.append(bc.item())

        if self.epoch % 1000 == 0:
            print('epoch %d, Loss: %.4e, Loss_mse: %.4e, Loss_pde: %.4e'
                  % (self.epoch, loss.item(), self.lw_mse * mse.item(), self.lw_pde * pde.item()))

        if self.epoch % 5000 == 0:
            bc_pred_test = self.predict_bc(self.in_test)
            bc_pred_test = bc_pred_test.detach().cpu().numpy()
            r2 = r2_score(self.out_test, np.squeeze(bc_pred_test))
            l2_error = np.linalg.norm(self.out_test - np.squeeze(bc_pred_test), 2) \
                       / np.linalg.norm(self.out_test, 2)
            print("one sample bc r2: ", r2, " l2_error: ", l2_error)

            fig = plt.figure(figsize=[4, 3])
            in_test_np = self.in_test.detach().cpu().numpy()[:, 0]
            plt.plot(in_test_np, self.out_test)
            plt.plot(in_test_np, bc_pred_test, "--")
            plt.savefig(f"training_results/iter_{self.epoch}.png")
            plt.close()

            out_pred_np = out_pred.detach().cpu().numpy()
            x_lin, y_lin = np.linspace(-1, 1, 21), np.linspace(-0.5, 0.5, 11)
            x_mesh, y_mesh = np.meshgrid(x_lin, y_lin)

            fig = plt.figure(figsize=[4, 3])
            plt.contourf(x_mesh, y_mesh, out_pred_np[-231:].reshape(11, 21), cmap="jet", levels=11)
            plt.colorbar()
            plt.savefig(f"training_results/u_iter_{self.epoch}.png")
            plt.close()

        return loss

    def train(self, epochs=10000):
        start_time = time.time()
        for i in range(self.epoch, epochs + 1):
            self.optimizer.step(self.loss)

        end_time = time.time()
        print(f"模型训练时间： {end_time - start_time}")

    def predict_u(self, inputs):
        predict_out = self.net_u(inputs)
        predict_out = predict_out.detach().cpu().numpy()

        return predict_out
