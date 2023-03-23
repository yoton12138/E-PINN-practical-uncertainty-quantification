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

    u_obs = u_real.reshape(-1, 1) + np.random.randn(u_real.shape[0], 1)*Opt.noise  # noise***********

    # boundary condition collocations
    x_bc_lin = np.linspace(-1, 1, 200)
    t_bc_lin = np.ones_like(x_bc_lin)
    flux_test = np.cos(0.5 * 3.1415926 * x_bc_lin)
    coords_bc_bottom = np.vstack((x_bc_lin, np.zeros_like(x_bc_lin) - 0.5, t_bc_lin)).T
    coords_bc_up = np.vstack((x_bc_lin, np.zeros_like(x_bc_lin) + 0.5, t_bc_lin)).T

    t_bc_lin = np.linspace(0, 1, 100)
    coords_bc_left = np.vstack((np.zeros_like(t_bc_lin) - 1, np.zeros_like(t_bc_lin) - 0.5, t_bc_lin)).T
    coords_bc_right = np.vstack((np.zeros_like(t_bc_lin) + 1, np.zeros_like(t_bc_lin) - 0.5, t_bc_lin)).T
    coords_bc_left_right = np.vstack((coords_bc_left, coords_bc_right))

    # pde grid collocations
    ub = np.array([1, -0.3, 1])
    lb = np.array([-1, -0.5, 0])
    coords_f = lb + (ub - lb)*lhs(3, 10000)  # 配点10000个
    coords_f = np.vstack([coords_f, coords_obs])

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

    model = MCDropout(layers_u, coords_obs_T, u_obs_T, coords_f_T, coords_bc_left_right_T, coords_bc_bottom_T, flux_test)
    model.train(epochs)

    end_time = time.time()
    print("耗时：", end_time - start_time)

    pred_list = []
    for i in range(100):
        q = model.predict_bc(coords_bc_bottom_T).data.cpu().numpy()
        pred_list.append(q)

    mu = np.mean(pred_list, axis=0)
    std = np.std(pred_list, axis=0)

    lower, upper = (mu - std * 2), (mu + std * 2)
    r2 = r2_score(flux_test, mu)
    rl2_error = np.linalg.norm(flux_test - np.squeeze(mu), 2) \
               / np.linalg.norm(flux_test, 2)
    print("Ensemble mean R2: ", r2, "rl2: ", rl2_error)

    loss_list = model.loss_list
    loss_pde = model.loss_pde
    loss_bc = model.loss_bc

    # 图
    experiment_tag = "_E-PINN_Adv_m5_0.01" # 便捷更改Pdf图等的名字
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
    plt.plot(loss_pde, label="pde loss")
    plt.plot(loss_list, label="total")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig("training_results/loss" + experiment_tag + ".png", dpi=600)