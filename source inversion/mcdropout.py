import numpy as np
import time
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class PINN(object):
    def __init__(self, layers_size_u, layer_size_s, in_obs, out_obs, collocation_points,
                 in_test=None, out_test=None, x_bc=None, y_bc=None):
        self.net_s = Fnn(layer_size_s, st=True, dropout=True).to(device)
        self.net_u = Fnn(layers_size_u).to(device)
        self.in_obs = in_obs
        self.out_obs = out_obs
        self.collocation_points = collocation_points

        self.in_test = in_test
        self.out_test = out_test
        self.x_bc = x_bc
        self.y_bc = y_bc

        self.pre_loss_u = 1.0

        self.mse_loss = torch.nn.MSELoss()
        self.nll_loss = torch.nn.GaussianNLLLoss()

        self.lw_mse = 1.
        self.lw_pde = 1.
        self.lw_bc = 0.0

        self.lr_u = 0.003
        self.lr_s = 0.005

        self.optimizer = torch.optim.Adam([
                                           {"params": self.net_u.parameters(), "lr": self.lr_u},
                                           {"params": self.net_s.parameters(), "lr": self.lr_s}
                                          ], betas=(0.9, 0.999), eps=1e-08)

        self.optimizer_u = torch.optim.Adam(self.net_u.parameters(),
                                            lr=self.lr_u, betas=(0.9, 0.999), eps=1e-8)

        self.epoch = 0
        self.loss_list = []

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
            l2_error = np.linalg.norm(self.out_test - np.squeeze(s_pred_test), 2) / np.linalg.norm(self.out_test, 2)
            print("sample r2: ", r2, " l2_error: ", l2_error)

        return loss

    def train(self, epochs=10000):
        for i in range(0, epochs + 1):
            self.optimizer.step(self.loss)


    def predict(self, x_test, M=100):
        """利用M个模型加权输出
        """
        pred_list = []
        for i in range(M):
            s_pred = self.net_s(x_test)
            s_pred_np = s_pred.data.cpu().numpy()
            pred_list.append(s_pred_np)

        return pred_list


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


def source_func(x):
    s1 = 1.0 * np.exp(-0.5 * ((x[:, 0] - 0.3)**2 + (x[:, 1] - 0.4)**2) / 0.15**2)

    return s1


def source_func_2(x):
    s1 = 1.0 * np.exp(-0.5 * ((x[:, 0] - 0.3)**2 + (x[:, 1] - 0.4)**2) / 0.15**2)
    s2 = 2.0 * np.exp(-0.5 * ((x[:, 0] - 0.8)**2 + (x[:, 1] - 0.8)**2) / 0.05**2)

    return s1 + s2


if __name__ == "__main__":
    # 观测数据准备
    noise_u = 0.02
    N_u = 2000
    solution = np.squeeze(loadmat("Matlab file/case_field_2/u.mat")["u"])
    positions = np.squeeze(loadmat("Matlab file/case_field_2/position.mat")["p"])
    index = np.random.randint(0, 2583, N_u)  # 4960 triangle
    u_star = solution[index].reshape(-1, 1) + np.random.randn(N_u, 1) * noise_u
    x_star = positions[0, :][index]
    y_star = positions[1, :][index]
    X_star = np.vstack([x_star, y_star]).T

    # 配点准备
    # 随机配点
    # N_f = 1000
    # lb = np.array([0., 0.])
    # ub = np.array([1.0, 1.0])
    # X_f = lb + (ub - lb) * lhs(2, N_f)
    # 网格配点
    nx = 50
    ny = 50
    xx = np.linspace(0, 1, nx)
    yy = np.linspace(0, 1, ny)
    xxx, yyy = np.meshgrid(xx, yy)
    X_f = np.hstack([xxx.flatten()[:, None], yyy.flatten()[:, None]])
    index_hole = np.where((X_f[:, 0] < 0.7) & (X_f[:, 0] > 0.5) & (X_f[:, 1] < 0.7) & (X_f[:, 1] > 0.5))[0]
    X_f_ = np.delete(X_f, index_hole, axis=0)
    collocations = np.vstack([X_f_, X_star])
    # bc
    bc_linspace = np.linspace(0.5, 0.7, 21)
    bc_grid_x,  bc_grid_y = np.meshgrid(bc_linspace, bc_linspace)
    bc_f = np.hstack([bc_grid_x.flatten()[:, None], bc_grid_y.flatten()[:, None]])
    index_hole_x = np.where((bc_f[:, 0] == 0.7) | (bc_f[:, 0] == 0.5))[0]
    index_hole_y = np.where((bc_f[:, 1] == 0.7) | (bc_f[:, 1] == 0.5))[0]
    bc_f_x = bc_f[index_hole_x]  # x 固定，x
    bc_f_y = bc_f[index_hole_y]
    # 观察对角线
    x_diag_0 = np.linspace(0, 0.5, 101)
    x_diag_1 = np.linspace(0.7, 1.0, 101)
    xy_diag_0 = np.vstack([x_diag_0, x_diag_0]).T
    xy_diag_1 = np.vstack([x_diag_1, x_diag_1]).T

    # source 场
    source_field_total = source_func_2(X_f)
    source_field_delete = source_func_2(X_f_)

    # numpy to tensor
    X_star_T = torch.tensor(X_star, dtype=torch.float32, requires_grad=True).to(device)
    u_star_T = torch.tensor(u_star, dtype=torch.float32).to(device)
    X_f_T = torch.tensor(X_f_, dtype=torch.float32, requires_grad=True).to(device)
    X_f_T_total = torch.tensor(X_f, dtype=torch.float32, requires_grad=True).to(device)
    collocations_T = torch.tensor(collocations, dtype=torch.float32, requires_grad=True).to(device)
    bc_f_x_T = torch.tensor(bc_f_x, dtype=torch.float32, requires_grad=True).to(device)
    bc_f_y_T = torch.tensor(bc_f_y, dtype=torch.float32, requires_grad=True).to(device)
    xy_diag_0_T = torch.tensor(xy_diag_0, dtype=torch.float32, requires_grad=True).to(device)
    xy_diag_1_T = torch.tensor(xy_diag_1, dtype=torch.float32, requires_grad=True).to(device)

    # 网络相关
    """
    构造一个神经网络Net_u拟合场数据，一个神经网络Net_s拟合热源
    """
    start_time = time.time()
    epochs = 100000
    layers_s = [2, 20, 20, 20, 20, 1]
    layers_u = [2, 20, 20, 20, 20, 1]

    PINNs = PINN(layers_u, layers_s, X_star_T, u_star_T, collocations_T, in_test=X_f_T, out_test=source_field_delete,
                 x_bc=bc_f_x_T, y_bc=bc_f_y_T)
    PINNs.train(epochs)
    end_time = time.time()
    print("耗时：", end_time - start_time)

    # 绘图
    plt.style.use("default")
    plt.rcParams['font.family'] = 'Times New Roman'
    save = True
    experiment_tag = "_DE_dropout0.1_eps0.0_noise_0.02"
    # 数值结果
    # x_mask = np.array([0.5, 0.5, 0.7, 0.7])  # 点有顺序，多边形填充
    # y_mask = np.array([0.5, 0.7, 0.7, 0.5])
    # fig, ax = plt.subplots(figsize=[4, 3], dpi=300)
    # #ax.plot(positions[0, :], positions[1, :], 'o', markersize=2, color='grey')
    # tpc = ax.tripcolor(positions[0, :], positions[1, :], solution, cmap="hot", shading="gouraud")
    # ax.fill(x_mask, y_mask, "white")
    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 1])
    # cb = plt.colorbar(tpc)
    # plt.show()
    # if save:
    #     plt.savefig("Numerical.png", bbox_inches="tight", dpi=300)
    #
    # # 热源场
    # mask = np.zeros_like(source_field_total, dtype=bool)
    # mask[index_hole] = True
    # source_field_masked = np.ma.array(source_field_total.reshape(nx, ny), mask=mask.reshape(nx, ny))
    #
    # fig, ax = plt.subplots(figsize=[4, 3], dpi=300)
    # #ax.plot(positions[0, :], positions[1, :] , 'o', markersize=2, color='grey')
    # ctf = ax.contourf(xxx, yyy, source_field_masked, cmap="hot", corner_mask=False, levels=100)
    # #ax.contour(xxx, yyy, source_field_masked, colors="k", linestyles="dash", corner_mask=False)
    # ax.fill(x_mask, y_mask, "white")  # 统一遮罩
    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 1])
    # cb = plt.colorbar(ctf)
    # #cb.set_ticks(np.array([0, 0.2, 0.4, 0.6, 0.8, 1]))
    # plt.show()
    # if save:
    #     plt.savefig("source.png", bbox_inches="tight", dpi=300)

    # 预测热源图
    s_pred_list = PINNs.predict(X_f_T_total, M=100)
    mu = np.mean(s_pred_list, axis=0)
    std = np.std(s_pred_list, axis=0)
    s_pred_mean_plot = np.delete(mu, index_hole, axis=0)
    s_r2 = r2_score(source_field_delete, s_pred_mean_plot)

    l2_error = np.linalg.norm(source_field_delete.reshape(-1, 1) - np.mean(s_pred_mean_plot, axis=0), 2) / np.linalg.norm(source_field_delete, 2)
    r2 = r2_score(source_field_delete, s_pred_mean_plot)
    print("Ensemble r2:", r2, "---l2_error:", l2_error)
    #
    mask = np.zeros_like(mu, dtype=bool)
    mask[index_hole] = True
    mu_masked = np.ma.array(mu.reshape(nx, ny), mask=mask.reshape(nx, ny))
    std_masked = np.ma.array(std.reshape(nx, ny), mask=mask.reshape(nx, ny))
    #
    fig, ax = plt.subplots(figsize=[4, 3])
    ctf0 = ax.contourf(xxx, yyy, mu_masked, cmap="hot", corner_mask=False, levels=100)
    #ax.contour(xxx, yyy, source_field_masked, colors="k", linestyles="dash", corner_mask=False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    cb0 = plt.colorbar(ctf0, ax=ax)
    plt.show()
    # if save:
    #     plt.savefig("pred_s_mean" + experiment_tag + ".png", bbox_inches="tight", dpi=300)

    x_diag_0 = np.linspace(0, 0.5, 101)
    x_diag_1 = np.linspace(0.7, 1.0, 101)
    xy_diag_0 = np.vstack([x_diag_0, x_diag_0]).T
    xy_diag_1 = np.vstack([x_diag_1, x_diag_1]).T
    fig, ax = plt.subplots(figsize=[4, 3])
    ctf0 = ax.contourf(xxx, yyy, std_masked, cmap="hot", corner_mask=False, levels=100)
    #ax.contour(xxx, yyy, source_field_masked, colors="k", linestyles="dash", corner_mask=False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.plot(x_diag_0, x_diag_0, linestyle="--", color="white")
    ax.plot(x_diag_1, x_diag_1, linestyle="--", color="white")
    cb0 = plt.colorbar(ctf0, ax=ax)
    plt.show()
    # if save:
    #     plt.savefig("pred_s_std" + experiment_tag + ".png", bbox_inches="tight", dpi=300)
    #
    # # loss
    loss_list = PINNs.loss_list
    plt.style.use('seaborn')
    plt.rcParams['font.family'] = 'Times New Roman'
    fig2 = plt.figure(figsize=[4, 3])

    plt.plot(loss_list, label="MC_dropout", alpha=0.8)
    plt.semilogy()
    plt.legend(loc=1)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()
    # if save:
    #     plt.savefig("loss" + experiment_tag + ".png", dpi=300, bbox_inches="tight")
    #
    #
    # # 对角线
    diag_pred_list_0 = PINNs.predict(xy_diag_0_T, M=100)
    diag_pred_list_1 = PINNs.predict(xy_diag_1_T, M=100)

    mu_diag_0 = np.mean(diag_pred_list_0, axis=0)
    mu_diag_1 = np.mean(diag_pred_list_1, axis=0)
    std_diag_0 = np.std(diag_pred_list_0, axis=0)
    std_diag_1 = np.std(diag_pred_list_1, axis=0)
    lower_0 = mu_diag_0 - 2 * std_diag_0
    lower_1 = mu_diag_1 - 2 * std_diag_1
    upper_0 = mu_diag_0 + 2 * std_diag_0
    upper_1 = mu_diag_1 + 2 * std_diag_1
    true_diag_0 = source_func_2(xy_diag_0)
    true_diag_1 = source_func_2(xy_diag_1)
    #
    fig3 = plt.figure(figsize=[4, 3])
    plt.fill_between(x_diag_0, lower_0.flatten(), upper_0.flatten(), alpha=0.5, rasterized=True, label="Epistemic uncertainty", color="green")
    plt.fill_between(x_diag_1, lower_1.flatten(), upper_1.flatten(), alpha=0.5, rasterized=True, color="green")
    plt.plot(x_diag_0, true_diag_0, color="xkcd:orange", linestyle="-", linewidth=2, label="Truth")
    plt.plot(x_diag_1, true_diag_1, color="xkcd:orange", linestyle="-", linewidth=2)
    plt.plot(x_diag_0, mu_diag_0, color="xkcd:dark blue", linestyle="--", linewidth=2, label='Identification')
    plt.plot(x_diag_1, mu_diag_1, color="xkcd:dark blue", linestyle="--", linewidth=2)
    plt.vlines(0.5, ymin=0, ymax=1, color="black", linestyle="--")
    plt.vlines(0.7, ymin=0, ymax=1, color="black", linestyle="--")
    plt.legend(loc=2)
    plt.xlabel("x")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()
    # if save:
    #     plt.savefig("diag" + experiment_tag + ".png", bbox_inches="tight", dpi=300)
    #
    # # 保存数据
    # np.savez(experiment_tag + ".npz", total=s_pred_list, mean=mu, std=std,
    #          diag_0=diag_pred_list_0, diag_1=diag_pred_list_1)
