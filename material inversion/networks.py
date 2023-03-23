import matplotlib.pyplot as plt
import numpy as np
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
            out = F.softplus(out) + 0.1
        return out


class BNNNet(torch.nn.Module):
    def __init__(self, layers_size, activation_function=torch.nn.Tanh,
                 prior_weight_mu=0., prior_weight_sigma=0.2, last_layer_act=False):
        super().__init__()

        self.constraint = False
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
        if last_layer_act:
            act = torch.nn.Softplus()
            self.model.add_module("act_last", act)

    def forward(self, x):
        out = self.model(x) + 0.1
        return out


class PINN(object):
    def __init__(self, layers_size, in_obs, out_obs, collocation_points, param,
                 in_test=None, out_test=None, xt_bc=None):
        self.net_u = Fnn(layers_size).to(Opt.device)
        self.in_obs = in_obs
        self.out_obs = out_obs
        self.collocation_points = collocation_points
        self.net_v = param
        self.in_test = in_test
        self.out_test = out_test
        self.xt_bc = xt_bc

        self.pre_loss_u = 1.0
        self.pre_loss_v = 1.0

        self.mse_loss = torch.nn.MSELoss()
        self.nll_loss = torch.nn.GaussianNLLLoss()

        self.lw_mse = Opt.lw_mse
        self.lw_pde = Opt.lw_pde
        self.lw_bc = Opt.lw_bc

        self.lr_u = Opt.lr_u
        self.lr_v = Opt.lr_v

        self.optimizer = torch.optim.Adam([
                                           {"params": self.net_u.parameters(), "lr": self.lr_u},
                                           {"params": self.net_v.parameters(), "lr": self.lr_v}
                                          ], betas=(0.9, 0.999), eps=1e-08)

        self.optimizer_u = torch.optim.Adam(self.net_u.parameters(),
                                            lr=self.lr_u, betas=(0.9, 0.999), eps=1e-8)
        self.optimizer_v = torch.optim.Adam(self.net_v.parameters(),
                                            lr=self.lr_v, betas=(0.9, 0.999), eps=1e-8)

        self.pre_epoch = 0
        self.epoch = 0
        self.M = Opt.M
        self.loss_list = []
        self.loss_m = []
        self.model_param_list = []

    def pde_loss(self):
        u, du_x, du_t, du_xx = Gradients(self.collocation_points, self.net_u, self.net_v).second_order()
        res = du_t - du_xx
        return torch.mean(res**2)

    def bc_loss(self):
        _, du_x, _ = Gradients(self.xt_bc, self.net_u, self.net_v).first_order()
        return torch.mean(du_x**2)

    def loss(self):
        out_pred = self.net_u(self.in_obs)
        mse = self.mse_loss(out_pred, self.out_obs)

        # adversarial training
        in_obs_at = self.fgm(mse, self.in_obs)
        out_pred_at = self.net_u(in_obs_at)
        mse_ = self.mse_loss(out_pred_at, self.out_obs)
        mse = mse + mse_
        #

        pde = self.pde_loss()
        bc_loss = self.bc_loss()

        loss = self.lw_mse*mse + self.lw_pde*pde + self.lw_bc*bc_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.epoch += 1
        self.loss_m.append(loss.item())

        if self.epoch % 1000 == 0:
            print('epoch %d, Loss: %.4e, Loss_mse: %.4e, Loss_pde: %.4e, Loss_bc: %.4e'
                  % (self.epoch, loss.item(), self.lw_mse*mse.item(), self.lw_pde*pde.item(),
                     self.lw_pde*bc_loss.item()))

        if self.epoch % 5000 == 0:
            v_pred_test = self.net_v(self.in_test.view(-1, 1))
            v_pred_test = v_pred_test.detach().cpu().numpy()
            r2 = r2_score(self.out_test, v_pred_test-0.1)
            l2_error = np.linalg.norm(self.out_test-0.1 - np.squeeze(v_pred_test), 2)\
                    / np.linalg.norm(self.out_test-0.1, 2)
            print("sample r2: ", r2, " l2_error: ", l2_error)

        return loss

    def loss_u(self):
        out_pred = self.net_u(self.in_obs)[:, 0].view(-1, 1)
        mse = self.mse_loss(out_pred, self.out_obs)
        loss = mse
        self.optimizer_u.zero_grad()
        loss.backward()
        self.pre_epoch += 1
        if self.pre_epoch % 1000 == 0:
            print('pre-epoch %d, pre_Loss_u: %.4e' % (self.pre_epoch, loss.item()))

        self.pre_loss_u = loss
        return loss

    def train(self, epochs=10000, pre_train=True, save_path="new_models/gp_at_models/"):
        for m in range(self.M):
            start_time = time.time()
            if pre_train:
                while self.pre_loss_u > 1e-4:
                    self.optimizer_u.step(self.loss_u)

            print("-----预训练结束-----")

            for i in range(self.epoch, epochs + 1):
                self.optimizer.step(self.loss)

            end_time = time.time()
            print(f"单个模型训练时间： {end_time - start_time}")

            stat_dict = self.net_v.state_dict()
            self.model_param_list.append(stat_dict)
            path = save_path
            if path:
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(stat_dict, path + "params_" + str(m))  # 保存之后用来Active sampling
            self.loss_list.append(self.loss_m)
            # 重新初始化一个模型
            self.net_u = Fnn(layers_size=Opt.layers_u).to(Opt.device)
            self.net_v = Fnn(layers_size=Opt.layers_v).to(Opt.device)
            self.optimizer = torch.optim.Adam([{"params": self.net_u.parameters(), "lr": self.lr_u},
                                               {"params": self.net_v.parameters(), "lr": self.lr_v}
                                               ], betas=(0.9, 0.999), eps=1e-08)
            self.optimizer_u = torch.optim.Adam(self.net_u.parameters(),
                                                lr=self.lr_u, betas=(0.9, 0.999), eps=1e-8)
            self.pre_loss_u = 1.0
            self.pre_epoch = 0
            self.epoch = 0
            self.loss_m = []

    def predict(self, x_test):
        """利用M个模型加权输出
        """
        model = Fnn(layers_size=Opt.layers_v).to(Opt.device)
        pre_list = []
        for param in self.model_param_list:
            model.load_state_dict(param)
            v_pre = model(x_test.view(-1, 1)).data.cpu().numpy()
            pre_list.append(v_pre)
        mean = np.mean(pre_list, axis=0)
        std = np.std(pre_list, axis=0)

        return mean, std

    def predict_m(self, x_test, m=0):
        """某个个模型输出
        """
        model = Fnn(layers_size=Opt.layers_v).to(Opt.device)
        param = self.model_param_list[m]
        model.load_state_dict(param)
        v_pre = model(x_test.view(-1, 1)).data.cpu().numpy()

        return v_pre

    def fgm(self, l, inputs, eps=0.0002):
        dl_X = torch.autograd.grad(l, inputs, torch.ones_like(l), retain_graph=True, create_graph=True)[0]
        norms = torch.norm(dl_X, 2, dim=1).view(-1, 1)  # 这种方式快,也需要把0和nan给替换成大数,保持摄动极小
        norms = torch.where(torch.isnan(norms), torch.full_like(norms, 1e16), norms)
        norms = torch.where(norms == 0, torch.full_like(norms, 1e16), norms)
        r_at = eps*dl_X/norms
        X_at = inputs + r_at

        return X_at


class Gradients(object):
    def __init__(self, inputs, net_u, net_v):
        self.inputs = inputs
        self.outs = net_u(inputs)
        self.net_u = net_u
        self.net_v = net_v

    def first_order(self):
        """输出有必要的一阶导数"""
        du_X = torch.autograd.grad(self.outs, self.inputs, torch.ones_like(self.outs),
                                   retain_graph=True, create_graph=True)[0]
        du_x = du_X[:, 0].view(-1, 1)
        du_t = du_X[:, 1].view(-1, 1)

        return self.outs, du_x, du_t

    def second_order(self):
        """输出有必要的二阶导数"""
        _, du_x, du_t = self.first_order()
        x = self.inputs[:, 0].view(-1, 1)
        vx = self.net_v(x)
        du_xX = torch.autograd.grad(vx*du_x, self.inputs, torch.ones_like(du_x),
                                    retain_graph=True, create_graph=True)[0]
        du_xx = du_xX[:, 0].view(-1, 1)  # 带有非线性项

        return self.outs, du_x, du_t, du_xx


class BPINN(object):
    def __init__(self, layers_size, in_obs, out_obs, collocation_points, param_v,
                 in_test=None, out_test=None, out_prior=None, xt_bc=None):
        self.net_u = Fnn(layers_size).to(Opt.device)  # 使用不同的 loss 1 2 对应不同的u的结构
        self.in_obs = in_obs
        self.out_obs = out_obs
        self.collocation_points = collocation_points
        self.net_v = param_v
        self.in_test = in_test
        self.out_test = out_test
        self.out_prior = out_prior
        self.xt_bc = xt_bc

        self.flag = False
        self.pre_loss_u = 1.0
        self.pre_loss_v = 1.0

        self.mse_loss = torch.nn.MSELoss()
        self.kl_loss = bnn.BKLLoss(reduction="mean")

        self.lw_mse = Opt.lw_mse
        self.lw_pde = Opt.lw_pde
        self.lw_bc = Opt.lw_bc
        self.lw_kl = Opt.lw_kl

        self.lr_u = Opt.lr_u
        self.lr_v = Opt.lr_v

        self.optimizer = torch.optim.Adam([{"params": self.net_u.parameters(), "lr": self.lr_u},
                                           {"params": self.net_v.parameters(), "lr": self.lr_v}],
                                          betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)

        self.optimizer_u = torch.optim.Adam(self.net_u.parameters(),
                                            lr=self.lr_u, betas=(0.9, 0.999), eps=1e-8)
        self.optimizer_v = torch.optim.Adam(self.net_v.parameters(),
                                            lr=self.lr_v, betas=(0.9, 0.999), eps=1e-8)

        self.epoch = 0
        self.loss_list = []

    def pde_loss(self):
        u, du_x, du_t, du_xx = Gradients(self.collocation_points, self.net_u, self.net_v).second_order()
        res = du_t - du_xx
        return torch.mean(res ** 2)

    def bc_loss(self):
        _, du_x, _ = Gradients(self.xt_bc, self.net_u, self.net_v).first_order()
        return torch.mean(du_x ** 2)

    def loss_u(self):
        out_pred = self.net_u(self.in_obs)
        mse = self.mse_loss(out_pred, self.out_obs)
        loss = mse
        self.optimizer_u.zero_grad()
        loss.backward()
        self.epoch += 1
        if self.epoch % 1000 == 0:
            print('epoch %d, pre_Loss_u: %.4e' % (self.epoch, loss.item()))

        self.pre_loss_u = loss
        return loss

    def loss_v(self):
        v_pred = self.net_v(self.in_test.view(-1, 1))
        mse = self.mse_loss(v_pred, self.out_prior)
        loss = mse
        self.optimizer_v.zero_grad()
        loss.backward()
        self.epoch += 1
        if self.epoch % 1000 == 0:
            print('epoch %d, pre_Loss_v: %.4e' % (self.epoch, loss.item()))

        self.pre_loss_v = loss
        return loss

    def loss_1(self):
        """只有v是BNN的损失函数"""
        out_pred = self.net_u(self.in_obs)
        mse = self.mse_loss(out_pred, self.out_obs)
        kl = self.kl_loss(self.net_v)
        pde = self.pde_loss()

        bc_loss = self.bc_loss()

        loss = self.lw_mse*mse + self.lw_pde*pde + self.lw_bc*bc_loss + self.lw_kl*kl

        self.optimizer.zero_grad()
        loss.backward()
        self.epoch += 1
        self.loss_list.append(loss.detach().cpu().numpy())

        if self.epoch % 1000 == 0:
            print('epoch %d, Loss: %.4e, Loss_mse: %.4e, Loss_kl: %.4e, Loss_pde: %.4e, Loss_bc: %.4e'
                  % (self.epoch, loss.item(), self.lw_mse*mse.item(), self.lw_kl*kl.item(), self.lw_pde*pde.item()
                     , self.lw_pde*bc_loss.item()))

        if self.epoch % 5000 == 0:
            v_pred_test = self.net_v(self.in_test.view(-1, 1))
            v_pred_test = v_pred_test.detach().cpu().numpy()
            r2 = r2_score(self.out_test, v_pred_test)
            l2_error = np.linalg.norm(self.out_test - np.squeeze(v_pred_test), 2) / np.linalg.norm(self.out_test, 2)
            print("one sample r2: ", r2, " l2_error: ", l2_error)

            fig = plt.figure(figsize=[4, 3])
            plt.plot(self.in_test.detach().cpu().numpy(), self.out_test)
            plt.plot(self.in_test.detach().cpu().numpy(), v_pred_test)
            plt.savefig(f"training_results/v_iter_{self.epoch}.png", dpi=600)
            plt.close()

            x_obs = np.linspace(0, 1, 13)
            t_obs = np.linspace(0.01, 0.03, 9)
            X_obs, T_obs = np.meshgrid(x_obs, t_obs)
            u_pred_test = self.net_u(self.in_obs)
            u_pred_test_np = u_pred_test.detach().cpu().numpy()
            fig = plt.figure(figsize=[4, 3])
            plt.contourf(X_obs, T_obs, u_pred_test_np.reshape(9, 13))
            plt.tight_layout()
            plt.savefig(f"training_results/u_iter_{self.epoch}.png", dpi=600)
            plt.close()

        return loss

    def train(self, epochs, pre_train=True):
        if pre_train:
            while self.pre_loss_u > 1e-4:
                self.optimizer_u.step(self.loss_u)

            while self.pre_loss_v > 1e-4 and self.epoch < 50000:
                self.optimizer_v.step(self.loss_v)

        self.epoch = 0

        for i in range(self.epoch, epochs + 1):
            self.optimizer.step(self.loss_1)

    def predict_u(self, inputs):
        predict_out = self.net_u(inputs)
        predict_out = predict_out.detach().cpu().numpy()

        return predict_out


class MCDropout(object):
    def __init__(self, layers_size, in_obs, out_obs, collocation_points, param,
                 in_test=None, out_test=None, xt_bc=None):
        self.net_u = Fnn(layers_size).to(Opt.device)
        self.in_obs = in_obs
        self.out_obs = out_obs
        self.collocation_points = collocation_points
        self.net_v = param
        self.in_test = in_test
        self.out_test = out_test
        self.xt_bc = xt_bc

        self.pre_loss_u = 1.0

        self.mse_loss = torch.nn.MSELoss()
        self.nll_loss = torch.nn.GaussianNLLLoss()

        self.lw_mse = Opt.lw_mse
        self.lw_pde = Opt.lw_pde
        self.lw_bc = Opt.lw_bc

        self.lr_u = Opt.lr_u
        self.lr_v = Opt.lr_v

        self.optimizer = torch.optim.Adam([
                                           {"params": self.net_u.parameters(), "lr": self.lr_u},
                                           {"params": self.net_v.parameters(), "lr": self.lr_v}
                                          ], betas=(0.9, 0.999), eps=1e-08)

        self.optimizer_u = torch.optim.Adam(self.net_u.parameters(),
                                            lr=self.lr_u, betas=(0.9, 0.999), eps=1e-8)

        self.pre_epoch = 0
        self.epoch = 0
        self.loss_list = []
        self.loss_m = []
        self.model_param_list = []

    def pde_loss(self):
        u, du_x, du_t, du_xx = Gradients(self.collocation_points, self.net_u, self.net_v).second_order()
        res = du_t - du_xx
        return torch.mean(res**2)

    def bc_loss(self):
        _, du_x, _ = Gradients(self.xt_bc, self.net_u, self.net_v).first_order()
        return torch.mean(du_x**2)

    def loss(self):
        out_pred = self.net_u(self.in_obs)
        mse = self.mse_loss(out_pred, self.out_obs)

        pde = self.pde_loss()
        bc_loss = self.bc_loss()

        loss = self.lw_mse*mse + self.lw_pde*pde + self.lw_bc*bc_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.epoch += 1
        self.loss_list.append(loss.item())

        if self.epoch % 1000 == 0:
            print('epoch %d, Loss: %.4e, Loss_mse: %.4e, Loss_pde: %.4e, Loss_bc: %.4e'
                  % (self.epoch, loss.item(), self.lw_mse*mse.item(), self.lw_pde*pde.item(),
                     self.lw_pde*bc_loss.item()))

        if self.epoch % 5000 == 0:
            v_pred_test = self.net_v(self.in_test.view(-1, 1))
            v_pred_test = v_pred_test.detach().cpu().numpy()
            r2 = r2_score(self.out_test, np.squeeze(v_pred_test))
            l2_error = np.linalg.norm(self.out_test - np.squeeze(v_pred_test), 2) / np.linalg.norm(self.out_test, 2)
            print("sample r2: ", r2, " l2_error: ", l2_error)

            fig = plt.figure(figsize=[4, 3])
            plt.plot(self.in_test.detach().cpu().numpy(), self.out_test)
            plt.plot(self.in_test.detach().cpu().numpy(), v_pred_test)
            plt.savefig(f"training_results/v_iter_{self.epoch}.png", dpi=600)
            plt.close()

            x_obs = np.linspace(0, 1, 13)
            t_obs = np.linspace(0.01, 0.03, 9)
            X_obs, T_obs = np.meshgrid(x_obs, t_obs)
            u_pred_test = self.net_u(self.in_obs)
            u_pred_test_np = u_pred_test.detach().cpu().numpy()
            fig = plt.figure(figsize=[4, 3])
            plt.contourf(X_obs, T_obs, u_pred_test_np.reshape(9, 13))
            plt.tight_layout()
            plt.savefig(f"training_results/u_iter_{self.epoch}.png", dpi=600)
            plt.close()

        return loss

    def loss_u(self):
        out_pred = self.net_u(self.in_obs)[:, 0].view(-1, 1)
        mse = self.mse_loss(out_pred, self.out_obs)
        loss = mse
        self.optimizer_u.zero_grad()
        loss.backward()
        self.pre_epoch += 1
        if self.pre_epoch % 1000 == 0:
            print('pre-epoch %d, pre_Loss_u: %.4e' % (self.pre_epoch, loss.item()))

        self.pre_loss_u = loss
        return loss

    def train(self, epochs=10000, pre_train=True):
        if pre_train:
            while self.pre_loss_u > 1e-4:
                self.optimizer_u.step(self.loss_u)

        print("-----预训练结束-----")

        for i in range(self.epoch, epochs + 1):
            self.optimizer.step(self.loss)

    def predict(self, x_test, M=100):
        """利用M个模型加权输出
        """
        pre_list = []
        for i in range(100):
            v_pre = self.net_v(x_test.view(-1, 1)).data.cpu().numpy()
            v_pre = np.log(v_pre - 0.1)  # 看情况是否需要log
            pre_list.append(v_pre)

        return pre_list

    def predict_u(self, inputs):
        predict_out = self.net_u(inputs)
        predict_out = predict_out.detach().cpu().numpy()

        return predict_out
    