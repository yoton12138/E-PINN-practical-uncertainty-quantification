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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, max_error
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
        self.net_v = Fnn(layers_size_v).to(device)
        self.net_u = Fnn(layers_size_u).to(device)
        self.in_obs = in_obs
        self.out_obs = out_obs
        self.collocation_points = collocation_points
        self.in_test = in_test
        self.out_test = out_test
        self.xt_bc = xt_bc

        #
        self.in_const_v = torch.tensor([]).to(device)
        self.out_const_v = torch.tensor([]).to(device)

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
        self.M = 5
        self.loss_list = []
        self.loss_m = []
        self.v_param_list = [0, 0, 0, 0, 0]
        self.u_param_list = [0, 0, 0, 0, 0]
        self.optimizer_list = [0, 0, 0, 0, 0]

        self.a_s = False
        self.is_stop = False
        self.expect_r2 = 0.99
        self.ensemble_r2 = 0.
        self.loss_as_list = []
        self.std_list = []
        self.mu_list = []
        self.r2_l2_list = []

    def pde_loss(self):
        u, du_x, du_t, du_xx = Gradients(self.collocation_points, self.net_u, self.net_v).second_order()
        res = du_t - du_xx
        return torch.mean(res**2)

    def bc_loss(self):
        _, du_x, _ = Gradients(self.xt_bc, self.net_u, self.net_v).first_order()
        return torch.mean(du_x**2)

    def loss_constrain(self):
        v_pred = self.net_v(self.in_const_v.view(-1, 1))
        loss = self.mse_loss(v_pred, self.out_const_v.view(-1, 1))
        return loss

    def loss(self):
        out_pred = self.net_u(self.in_obs)
        mse = self.mse_loss(out_pred, self.out_obs)

        # adversarial training
        in_obs_at = self.fgm(mse, self.in_obs)
        out_pred_at = self.net_u(in_obs_at)
        mse_ = self.mse_loss(out_pred_at, self.out_obs)
        mse = mse + mse_
        # active constraint
        if self.a_s:
            mse_c = self.loss_constrain()
            mse = mse + mse_c

        pde = self.pde_loss()
        bc_loss = self.bc_loss()

        loss = self.lw_mse*mse + self.lw_pde*pde + self.lw_pde*bc_loss

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
            r2 = r2_score(self.out_test, v_pred_test)
            l2_error = np.linalg.norm(self.out_test - v_pred_test, 2) / np.linalg.norm(self.out_test, 2)
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

    def train(self, epochs=10000, pre_train=True):
        for m in range(self.M):
            if pre_train:
                while self.pre_loss_u > 1e-4:
                    self.optimizer_u.step(self.loss_u)

            print("-----预训练结束-----")
            # normal training
            for i in range(epochs + 1):
                self.optimizer.step(self.loss)

            # save current model u and v
            stat_dict_u = self.net_u.state_dict()
            path = "models/gp_at_models_u/params_"
            #torch.save(stat_dict_u, path + str(m))
            stat_dict_v = self.net_v.state_dict()
            path = "models/gp_at_models_v/params_"
            #torch.save(stat_dict_v, path + str(m))

            self.loss_list.append(self.loss_m)
            self.u_param_list[m] = stat_dict_u
            self.v_param_list[m] = stat_dict_v
            self.optimizer_list[m] = self.optimizer

            # initial a new model for ensemble
            self.net_u = Fnn(layers_size=layers_u).to(device)
            self.net_v = Fnn(layers_size=layers_v).to(device)
            self.optimizer = torch.optim.Adam([{"params": self.net_u.parameters(), "lr": self.lr_u},
                                               {"params": self.net_v.parameters(), "lr": self.lr_v}
                                               ], betas=(0.9, 0.999), eps=1e-08)
            self.optimizer_u = torch.optim.Adam(self.net_u.parameters(),
                                                lr=self.lr_u, betas=(0.9, 0.999), eps=1e-8)
            self.pre_loss_u = 1.0
            self.pre_epoch = 0
            self.epoch = 0
            self.loss_m = []

        mu, std = self.predict(self.in_test)
        self.ensemble_r2 = r2_score(v_test, mu)
        self.std_list.append(std)
        self.mu_list.append(mu)
        self.is_stop = self.stop_criterion(std)

    def as_train(self, epochs=10000, path=None):
        """
        active sampling training
        :param epochs:
        :param path: list of u and v models
        :return: None
        """
        print("-----Active Sampling-----")
        if path:  # 增加一个可以通过路径来主动采样，方便调试
            for m in range(self.M):
                path_u = path[0] + "/params_" + str(m)
                path_v = path[1] + "/params_" + str(m)
                self.u_param_list[m] = torch.load(path_u)
                self.v_param_list[m] = torch.load(path_v)

        while not self.is_stop:
            self.a_s = True  # 开启约束损失
            self.active_sampling()
            print("-----Expect r2: ", self.expect_r2, "-----Current r2: ", self.ensemble_r2)
            self.is_stop = self.stop_criterion(self.std_list[-1])
            if self.is_stop:
                break  # 有一步的计算延迟，所以干脆再判断一下

            for m in range(self.M):
                self.net_u = Fnn(layers_size=layers_u).to(device)
                self.net_u.load_state_dict(self.u_param_list[m])
                self.net_v = Fnn(layers_size=layers_v).to(device)
                self.net_v.load_state_dict(self.v_param_list[m])

                self.optimizer = torch.optim.Adam([{"params": self.net_u.parameters(), "lr": self.lr_u},
                                                   {"params": self.net_v.parameters(), "lr": self.lr_v}
                                                   ], betas=(0.9, 0.999), eps=1e-08)
                self.epoch = 0
                self.loss_m = []

                for i in range(epochs + 1):
                    self.optimizer.step(self.loss)

                self.u_param_list[m] = self.net_u.state_dict()
                self.v_param_list[m] = self.net_v.state_dict()
                self.loss_as_list.append(self.loss_m)

            # 完成一轮主动采样之后，看看具体效果
            mu, std = self.predict(self.in_test)
            lower, upper = (mu - std * 2), (mu + std * 2)
            x_as = self.in_const_v.data.cpu().numpy()
            v_as = self.out_const_v.data.cpu().numpy()
            plt.style.use("seaborn")
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.figure(figsize=[4, 3])
            plt.fill_between(x_test, lower.flatten(), upper.flatten(), alpha=0.5, rasterized=True,
                             label="Epistemic uncertainty")
            plt.plot(x_test, np.log(v_test-0.1), color="xkcd:orange", linestyle="-", linewidth=2, label="Truth")
            plt.plot(x_test, mu, color="xkcd:dark blue", linestyle="--", linewidth=2, label='Identification')
            plt.scatter(x_as, np.log(v_as-0.1), color="green", marker="+", label="Active sample")
            plt.legend(loc=3)
            plt.xlabel("x")
            #plt.ylim([-1.25, 0.75])
            plt.ylim([-0.6, 0.6])
            plt.ylabel("log[v(x)-0.1]")
            plt.tight_layout()
            #plt.savefig("logvx_" + str(self.in_const_v.shape[0]) + ".pdf", bbox_inches="tight")
            plt.savefig("logvx_" + str(self.in_const_v.shape[0]) + ".png", bbox_inches="tight", dpi=300)

            m_gp = np.log(self.out_test - 0.1)
            r2 = round(r2_score(m_gp, mu), 4)
            l2_error = round(np.linalg.norm(np.squeeze(mu) - m_gp, 2) / np.linalg.norm(m_gp, 2), 4)
            print("Ensemble R2: ", r2, "--r_l2: ", l2_error)
            self.r2_l2_list.append(np.array([r2, l2_error]))

    def active_sampling(self):
        """
        训练好基本的模型之后开始主动采样提升模型
        """
        mu, std = self.predict(self.in_test)
        self.ensemble_r2 = r2_score(np.log(v_test-0.1), mu)  # 垃圾模型会报错，要注意 log是否有
        self.std_list.append(std)
        self.mu_list.append(mu)
        x_as_i_T = self.in_test[np.argmax(std)].expand(1)
        vx_as_i_T = torch.tensor(v_test[np.argmax(std)], dtype=torch.float32).expand(1).to(device)
        self.in_const_v = torch.cat([self.in_const_v, x_as_i_T], dim=0)
        self.out_const_v = torch.cat([self.out_const_v, vx_as_i_T], dim=0)

    def predict(self, x_test):
        """利用M个模型加权输出
        """
        model = Fnn(layers_size=layers_v).to(device)
        v_list = []
        for param in self.v_param_list:
            model.load_state_dict(param)
            v_pred = model(x_test.view(-1, 1)).data.cpu().numpy()
            v_pred = np.log(v_pred-0.1)  # 看情况是否需要log
            v_list.append(v_pred)
        mean = np.mean(v_list, axis=0)
        std = np.std(v_list, axis=0)

        return mean, std

    def predict_m(self, x_test, m=0):
        """某个个模型输出
        """
        model = Fnn(layers_size=layers_v).to(device)
        param = self.v_param_list[m]
        model.load_state_dict(param)
        v_pred = model(x_test.view(-1, 1)).data.cpu().numpy()

        return v_pred

    def fgm(self, l, inputs, eps=0.0002):
        dl_X = torch.autograd.grad(l, inputs, torch.ones_like(l), retain_graph=True, create_graph=True)[0]
        norms = torch.norm(dl_X, 2, dim=1).view(-1, 1)  # 这种方式快,也需要把0和nan给替换成大数,保持摄动极小
        norms = torch.where(torch.isnan(norms), torch.full_like(norms, 1e16), norms)
        norms = torch.where(norms == 0, torch.full_like(norms, 1e16), norms)
        r_at = eps*dl_X/norms
        X_at = inputs + r_at

        return X_at

    def stop_criterion(self, std, tolerance=0.05):
        """
        停止主动采样的标准, 响应区间的范围5%吧
        """
        range_v = self.mu_list[-1].max() - self.mu_list[-1].min()
        if std.max() > tolerance*range_v:
            return False
        else:
            return True


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
    #PINNs.train(epochs, pre_train=True)
    model_path = ["models/sin_at_models_u", "models/sin_at_models_v"]
    #PINNs.as_train(epochs=10000, path=model_path)
    end_time = time.time()
    print("耗时：", end_time - start_time)

    # 两种预测方式， 直接预测，
    # mu, std = Net_u.predict(x_test_T)

    # 转化log预测
    # logv_list = []
    # r2_list = []
    # for i in range(len(PINNs.v_param_list)):
    #     v_pred = PINNs.predict_m(x_test_T, m=int(i))
    #     logv = np.log(np.asarray(v_pred - 0.1))
    #     r2_i = r2_score(np.log(v_test - 0.1), logv)
    #     logv_list.append(logv)
    #     r2_list.append(r2_i)
    # mu = np.mean(logv_list, axis=0)
    # std = np.std(logv_list, axis=0)
    # lower, upper = (mu - std * 2), (mu + std * 2)
    # r2 = r2_score(np.log(v_test - 0.1), mu)
    # r2_list.append(r2)
    # print("Ensemble mean R2: ", r2)
    #
    # loss_list = PINNs.loss_list
    # loss_as_list = PINNs.loss_as_list
    # x_test_const = PINNs.in_const_v.data.cpu().numpy()[:-1]
    # v_test_const = PINNs.out_const_v.data.cpu().numpy()[:-1]
    #
    # # 自定义可视化标准单图回归不确定性图例模板
    # experiment_tag = "_DE_Adv_m5_0.01"  # 便捷更改Pdf图等的名字
    # save = True
    # # 保存一些数据
    # np.savez("logvx" + experiment_tag + ".npz", total=logv_list, mean=mu, std=std, loss_list=loss_list, r2_list=r2_list,
    #          r2_l2_list=PINNs.r2_l2_list)

    plt.style.use("seaborn")
    plt.rcParams['font.family'] = 'Times New Roman'
    # fig1 = plt.figure(figsize=[4, 3])
    # plt.fill_between(x_test, lower.flatten(), upper.flatten(), alpha=0.5, rasterized=True, label="Epistemic uncertainty")
    # plt.plot(x_test, np.log(v_test - 0.1), color="xkcd:orange", linestyle="-", linewidth=2, label="Truth")
    # plt.plot(x_test, mu, color="xkcd:dark blue", linestyle="--", linewidth=2, label='Identification')
    # plt.scatter(x_test_const, np.log(v_test_const-0.1), color="green", marker="+", label="Active sample")
    # plt.legend(loc=3)
    # plt.xlabel("x")
    # plt.ylabel("log[v(x)-0.1]")
    # plt.tight_layout()
    # plt.show()
    # if save:
    #     plt.savefig("vx" + experiment_tag + ".pdf", bbox_inches="tight")
    #
    # fig2 = plt.figure(figsize=[4, 3])
    # for i, loss_m in enumerate(loss_list):
    #     plt.plot(loss_m, label="model_" + str(i), alpha=0.1*(7-int(i)))
    # plt.semilogy()
    # plt.legend(loc=1)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.tight_layout()
    # plt.show()
    # if save:
    #     plt.savefig("loss" + experiment_tag + ".pdf", bbox_inches="tight")
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
    #
    # fig4 = plt.figure(figsize=[4, 3])
    # for i, loss_m in enumerate(loss_as_list[-5:]):
    #     plt.plot(loss_m, label="model_as_" + str(i), alpha=0.1*(7-int(i)))
    # plt.semilogy()
    # plt.legend(loc=1)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.tight_layout()
    # plt.show()
    # if save:
    #     plt.savefig("loss_as" + experiment_tag + ".pdf", bbox_inches="tight")
    #
    # fig5 = plt.figure(figsize=[4, 3])
    # plt.plot(x_test, std)
    # plt.legend()
    # plt.xlabel("x")
    # plt.ylabel("Standard deviation")
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("std-0.01" + experiment_tag + ".pdf", bbox_inches="tight")

    # 主动采样的图
    plt.style.use("seaborn-whitegrid")
    plt.rcParams['font.family'] = 'Times New Roman'
    x_lin = np.linspace(0, 8, 9)
    active_r2_list = [0.9336, 0.9896, 0.9916, 0.9971, 0.9976, 0.9984, 0.9984, 0.9995, 0.9995]
    active_l2_list = [0.2576, 0.1018, 0.0915, 0.0541, 0.0495, 0.0405, 0.0396, 0.0224, 0.0218]
    fig4 = plt.figure(figsize=[4, 3])
    plt.plot(x_lin, active_r2_list, linestyle="--", linewidth=2, marker="o",
             label='Identification')
    plt.xlabel("active sample numbers")
    plt.ylabel("R-square")
    plt.tight_layout()
    plt.show()
    plt.savefig("active_r2.png", bbox_inches="tight", dpi=300, transparent=True)

    fig5 = plt.figure(figsize=[4, 3])
    plt.plot(x_lin, active_l2_list, linestyle="--", linewidth=2, marker="o",
             label='Identification')
    plt.xlabel("active sample numbers")
    plt.ylabel("relative l2 error")
    plt.tight_layout()
    plt.show()
    plt.savefig("active_l2.png", bbox_inches="tight", dpi=300, transparent=True)
