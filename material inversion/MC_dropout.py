import numpy as np
import time
import os, sys, re
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pyDOE import lhs
import torchbnn as bnn
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


def seed_torch(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(1234)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using device: ", torch.cuda.get_device_name(0))


class Fnn(nn.Module):
    def __init__(self, layers_size, dropout=False):
        super().__init__()

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

        for i in range(len(h_layer_list) - 1):
            layer_name = "hidden_layer_" + str(i)
            self.model.add_module(layer_name, nn.Linear(h_layer_list[i], h_layer_list[i + 1]))
            self.model.add_module("ac_fun" + str(i), self.activation_func)
            if dropout:
                self.model.add_module("dropout_" + str(i), nn.Dropout(p=0.1))

        self.output = nn.Linear(h_layer_list[-1], o_layer_size)
        self.model.add_module("output_layer", self.output)

    def forward(self, x):
        out = self.model(x)

        return out


class PINN(object):
    def __init__(self, layers_size_u, layers_size_v, in_obs, out_obs, collocation_points,
                 in_test=None, out_test=None, xt_bc=None):
        self.net_v = Fnn(layers_size_v, dropout=True).to(device)
        self.net_u = Fnn(layers_size_u).to(device)  

        self.in_obs = in_obs
        self.out_obs = out_obs
        self.collocation_points = collocation_points
        self.in_test = in_test
        self.out_test = out_test
        self.xt_bc = xt_bc

        self.pre_loss_u = 1.0

        self.mse_loss = torch.nn.MSELoss()

        self.lw_mse = 1.
        self.lw_pde = 0.001

        self.lr_u = 0.003
        self.lr_v = 0.005

        self.optimizer = torch.optim.Adam([
                                           {"params": self.net_u.parameters(), "lr": self.lr_u},
                                           {"params": self.net_v.parameters(), "lr": self.lr_v}
                                          ], betas=(0.9, 0.999), eps=1e-08)

        self.optimizer_u = torch.optim.Adam(self.net_u.parameters(),
                                            lr=self.lr_u, betas=(0.9, 0.999), eps=1e-8)

        self.pre_epoch = 0
        self.epoch = 0
        self.loss_list = []
        
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

        loss = self.lw_mse*mse + self.lw_pde*pde + self.lw_pde*bc_loss

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
            r2 = r2_score(self.out_test, v_pred_test)
            l2_error = np.linalg.norm(self.out_test - np.squeeze(v_pred_test), 2) / \
                       np.linalg.norm(self.out_test, 2)
            print("--no log-- sample r2: ", r2, " l2_error: ", l2_error)

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
        for i in range(epochs + 1):
            self.optimizer.step(self.loss)

    def predict(self, x_test, M=100):
        """利用M个模型加权输出
        """
        pred_list = []
        for i in range(100):
            v_pre = self.net_v(x_test.view(-1, 1)).data.cpu().numpy()
            v_pre = np.log(v_pre - 0.1)  # 看情况是否需要log
            pred_list.append(v_pre)
        # mean = np.mean(pre_list, axis=0)
        # std = np.std(pre_list, axis=0)

        return pred_list


class Gradients(object):
    def __init__(self, inputs, net_u, net_v):
        self.inputs = inputs
        self.outs = net_u(inputs)
        self.net_u = net_u
        self.net_v = net_v

        self.L_flag = False
        self.L = None

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


def gp_(x, gp_data):
    x_gp = np.linspace(0, 1, 49)
    if np.where(x_gp == x)[0].size > 0:
        return np.exp(gp_data[np.where(x_gp == x)[0][0]]) + 0.1
    else:
        index = np.where(x_gp > x)[0][0]
        x1 = x_gp[index-1]
        x2 = x_gp[index]
        y1 = np.exp(gp_data[index-1]) + 0.1
        y2 = np.exp(gp_data[index]) + 0.1
        delta_x = x2 - x1
        y = ((x2 - x)/delta_x)*y1 + ((x - x1)/delta_x)*y2
        return y


if __name__ == "__main__":
    # 观测数据准备
    measures = loadmat("Matlab file/measures_sin_pure.mat")["measure"]
    solution = loadmat("Matlab file/solution_sin.mat")["sol"][101:, :]
    gp_data = np.squeeze(loadmat("Matlab file/gp_points.mat")["rf"])

    x_sol = np.linspace(0, 1, 49)
    t_sol = np.linspace(0.01, 0.03, 200)
    X_sol, T_sol = np.meshgrid(x_sol, t_sol)
    X_fem = np.hstack([X_sol.flatten()[:, None], T_sol.flatten()[:, None]])
    u_fem = solution.reshape(-1, 1)

    x_obs = np.linspace(0, 1, 13)
    t_obs = np.linspace(0.01, 0.03, 9)
    X_obs, T_obs = np.meshgrid(x_obs, t_obs)
    X_star = np.hstack([X_obs.flatten()[:, None], T_obs.flatten()[:, None]])
    u_star = measures.reshape(-1, 1) + np.random.randn(117, 1)*0.01  # noise***********

    x_test = np.linspace(0, 1, 409)
    v_test = np.exp(0.5*np.sin(2*np.pi*x_test)) + 0.1  # sin形式
    #v_test = np.array([gp_(x, gp_data) for x in x_test])  # gp形式
    mx = np.log(v_test - 0.1)

    X_BC, T_BC = np.meshgrid(np.array([0, 1.0]), t_sol)
    XT_BC = np.hstack([X_BC.flatten()[:, None], T_BC.flatten()[:, None]])

    # 配点准备
    # 随机配点
    # N_f = 1000
    # lb = np.array([0, 0.01])
    # ub = np.array([1.0, 0.03])
    # X_f = lb + (ub - lb)*lhs(2, N_f)
    # 网格配点
    nx = 50
    nt = 100
    xx = np.linspace(0, 1, nx+2)[1:-1]
    tt = np.linspace(0.01, 0.03, nt+2)[1:-1]
    xxx, ttt = np.meshgrid(xx, tt)
    X_f = np.hstack([xxx.flatten()[:, None], ttt.flatten()[:, None]])

    X_f = np.vstack([X_f, X_star])

    # numpy to tensor
    X_star_T = torch.tensor(X_star, dtype=torch.float32, requires_grad=True).to(device)
    u_star_T = torch.tensor(u_star, dtype=torch.float32).to(device)
    X_f_T = torch.tensor(X_f, dtype=torch.float32, requires_grad=True).to(device)
    x_test_T = torch.tensor(x_test, dtype=torch.float32).to(device)
    X_fem_T = torch.tensor(X_fem, dtype=torch.float32, requires_grad=True).to(device)
    u_fem_T = torch.tensor(u_fem, dtype=torch.float32).to(device)
    XT_BC_T = torch.tensor(XT_BC, dtype=torch.float32, requires_grad=True).to(device)
    # 网络相关
    """
    构造一个神经网络bNet_u拟合场数据，一个神经网络bNet_v拟合扩散率v
    """
    start_time = time.time()
    epochs = 50000
    layers_v = [1, 20, 20, 20, 20, 1]
    layers_u = [2, 20, 20, 20, 20, 1]
    
    PINNs = PINN(layers_u, layers_v, X_star_T, u_star_T, X_f_T, x_test_T, v_test, xt_bc=XT_BC_T)
    PINNs.train(epochs)
    end_time = time.time()
    print("耗时：", end_time - start_time)

    # 直接预测
    v_pre_list = PINNs.predict(x_test_T, M=100)
    mu = np.mean(v_pre_list, axis=0)
    std = np.std(v_pre_list, axis=0)
    lower, upper = (mu - std * 2), (mu + std * 2)
    r2 = r2_score(np.log(v_test - 0.1), mu)
    l2_error = np.linalg.norm(np.log(v_test - 0.1) - np.squeeze(mu), 2) / np.linalg.norm(np.log(v_test - 0.1), 2)
    print("mean R2: ", r2, "L2: ", l2_error)

    loss_list = PINNs.loss_list

    # 自定义可视化标准单图回归不确定性图例模板
    experiment_tag = "_dropout0.1_noise0.01"  # 便捷更改Pdf图等的名字
    save = True
    # 保存一些数据
    np.savez("logvx" + experiment_tag + ".npz", total=v_pre_list, mean=mu, std=std, loss_list=loss_list)

    plt.style.use("seaborn")
    plt.rcParams['font.family'] = 'Times New Roman'
    fig1 = plt.figure(figsize=[4, 3])
    plt.fill_between(x_test, lower.flatten(), upper.flatten(), alpha=0.5, rasterized=True, label="Epistemic uncertainty")
    plt.plot(x_test, np.log(v_test - 0.1), color="xkcd:orange", linestyle="-", linewidth=2, label="Truth")
    plt.plot(x_test, mu, color="xkcd:dark blue", linestyle="--", linewidth=2, label='Identification')
    plt.legend(loc=3)
    plt.xlabel("x")
    plt.ylabel("log[v(x)-0.1]")
    # plt.ylim([-1.25, 0.75])
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("vx" + experiment_tag + ".png", dpi=300)

    fig2 = plt.figure(figsize=[4, 3])

    plt.plot(loss_list, label="MC_dropout", alpha=0.8)
    plt.semilogy()
    plt.legend(loc=1)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("loss" + experiment_tag + ".png", dpi=300)
    #
    # fig3 = plt.figure(figsize=[4, 3])
    # plt.plot(x_test, np.log(v_test-0.1), color="xkcd:orange", linestyle="-", linewidth=2, label="Truth")
    # for i in range(5):
    #     v_pre = PINNs.predict_m(x_test_T.view(-1, 1), m=int(i))
    #     if i == 0:
    #         plt.plot(x_test, np.log(v_pre-0.1), color="xkcd:dark blue", alpha=0.3, linestyle="-", linewidth=2, label='Identification')
    #     else:
    #         plt.plot(x_test, np.log(v_pre-0.1), color="xkcd:dark blue", alpha=0.3, linestyle="-", linewidth=2)
    # plt.scatter(x_test_const, np.log(v_test_const-0.1), color="green", marker="+", label="Active sample")
    # plt.legend(loc=3)
    # plt.xlabel("x")
    # plt.ylabel("log[v(x)-0.1]")
    # plt.tight_layout()
    # plt.show()
    # if save:
    #     plt.savefig("samples" + experiment_tag + ".pdf", bbox_inches="tight")

