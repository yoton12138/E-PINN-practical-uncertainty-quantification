import matplotlib
import numpy as np

from networks import *
from options import Opt
from utils import *
import matplotlib.pyplot as plt
from pyDOE import lhs

matplotlib.use("agg")
seed_torch(1234)

if __name__ == "__main__":
    # 观测数据准备
    data = np.load("data/ExtractedDataCos.npz")
    coords_sensor = data["Nodes"][:, 1:3]
    coords_sensor = np.tile(coords_sensor, (51, 1))
    u_real = data["Temperature"].reshape(-1, 1)  # 231*51 的数据
    t_lin = np.linspace(0, 1, 51)
    t_lin = np.repeat(t_lin, 231).reshape(-1, 1)
    coords_obs = np.hstack((coords_sensor, t_lin))

    # 查看数据
    # plt.rcParams['font.family'] = 'Times New Roman'
    # x_lin, y_lin = np.linspace(-1, 1, 21), np.linspace(-0.5, 0.5, 11)
    # x_mesh, y_mesh = np.meshgrid(x_lin, y_lin)
    # fig = plt.figure(figsize=[6, 3])
    # plt.contourf(x_mesh, y_mesh, u_real[-231:].reshape(11, 21), cmap="coolwarm", levels=101)
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig("training_results/1.png", dpi=600)

    u_obs = u_real.reshape(-1, 1) + np.random.randn(u_real.shape[0], 1)*Opt.noise  # noise***********

    # boundary condition collocations
    x_bc_lin = np.linspace(-1, 1, 200)
    t_bc_lin = np.ones_like(x_bc_lin)  # 用区间后段的数据更准确
    flux_test = np.cos(0.5 * 3.1415926 * x_bc_lin)
    coords_bc_bottom = np.vstack((x_bc_lin, np.zeros_like(x_bc_lin) - 0.5, t_bc_lin)).T
    coords_bc_up = np.vstack((x_bc_lin, np.zeros_like(x_bc_lin) + 0.5, t_bc_lin)).T

    t_bc_lin = np.linspace(0, 1, 100)
    coords_bc_left = np.vstack((np.zeros_like(t_bc_lin) - 1, np.zeros_like(t_bc_lin) - 0.5, t_bc_lin)).T
    coords_bc_right = np.vstack((np.zeros_like(t_bc_lin) + 1, np.zeros_like(t_bc_lin) - 0.5, t_bc_lin)).T
    coords_bc_left_right = np.vstack((coords_bc_left, coords_bc_right))

    ub = np.array([1, -0.3, 1])
    lb = np.array([-1, -0.5, 0])
    coords_f = lb + (ub - lb)*lhs(3, 10000)  # 配点10000试试
    coords_f = np.vstack([coords_f, coords_obs])

    # fig2 = plt.figure(figsize=[6, 3])
    # arr_index = np.where(coords_f[:, 2] > 0.99)[0]
    # coords_f_plot = coords_f[arr_index]
    # plt.scatter(coords_f_plot[:, 0], coords_f_plot[:, 1], s=10, marker="+", color="black")
    # plt.tight_layout()
    # plt.savefig("training_results/2.png", dpi=600)

    # numpy to tensor
    coords_obs_T = torch.tensor(coords_obs, dtype=torch.float32, requires_grad=True).to(Opt.device)
    u_obs_T = torch.tensor(u_obs, dtype=torch.float32).to(Opt.device)
    coords_f_T = torch.tensor(coords_f, dtype=torch.float32, requires_grad=True).to(Opt.device)
    u_real_T = torch.tensor(u_real, dtype=torch.float32).to(Opt.device)
    coords_bc_bottom_T = torch.tensor(coords_bc_bottom, dtype=torch.float32, requires_grad=True).to(Opt.device)
    coords_bc_left_right_T = torch.tensor(coords_bc_left_right, dtype=torch.float32, requires_grad=True).to(Opt.device)

    # 网络相关
    """
    构造一个神经网络拟合场数据
    """
    start_time = time.time()
    epochs = Opt.epochs
    layers_u = Opt.layers_u

    model = PINN(layers_u, coords_obs_T, u_obs_T, coords_f_T, coords_bc_left_right_T, coords_bc_bottom_T, flux_test)
    model.train(epochs)
    # path_list = ["save_models/exp4/params_0", "save_models/exp4/params_1", "save_models/exp4/params_2",
    #              "save_models/exp4/params_3", "save_models/exp4/params_4"]
    # net_u.load_models(path_list)

    end_time = time.time()
    print("耗时：", end_time - start_time)

    # 预测
    mu, std = model.predict(coords_bc_bottom_T)

    lower, upper = (mu - std * 2), (mu + std * 2)
    r2 = r2_score(flux_test, mu)
    rl2_error = np.linalg.norm(flux_test - np.squeeze(mu), 2) \
               / np.linalg.norm(flux_test, 2)
    print("Ensemble mean R2: ", r2, "rl2: ", rl2_error)

    loss_list = model.loss_list

    # 图
    experiment_tag = "_E-PINN_Adv_m5_0.01"
    save = True
    # 保存一些数据
    np.savez("training_results/bc" + experiment_tag + ".npz", mean=mu, std=std, loss_list=loss_list)

    plt.style.use("seaborn")
    plt.rcParams['font.family'] = 'Times New Roman'
    fig1 = plt.figure(figsize=[4, 3])
    plt.fill_between(x_bc_lin, lower.flatten(), upper.flatten(), alpha=0.5, rasterized=True, label="Epistemic uncertainty")
    plt.plot(x_bc_lin, flux_test, color="xkcd:orange", linestyle="-", linewidth=2, label="Truth")
    plt.plot(x_bc_lin, mu, color="xkcd:dark blue", linestyle="--", linewidth=2, label='Identification')
    plt.legend(loc=4)
    plt.xlabel("x")
    plt.ylabel("q(x)")
    plt.tight_layout()
    if save:
        plt.savefig("training_results/bc" + experiment_tag + ".png", dpi=600)

    fig2 = plt.figure(figsize=[4, 3])
    for i, loss_m in enumerate(loss_list):
        plt.plot(loss_m, label="model_" + str(i), alpha=0.1*(7-int(i)))
    plt.semilogy()
    plt.legend(loc=1)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout()
    if save:
        plt.savefig("training_results/loss" + experiment_tag + ".png", dpi=600)