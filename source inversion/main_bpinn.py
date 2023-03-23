import matplotlib
import numpy as np

from Networks import *
from scipy.io import loadmat
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

matplotlib.use("Agg")
seed_torch(1234)

if __name__ == "__main__":
    # 观测数据准备
    noise_u = Opt.noise
    N_u = 2000
    solution = np.squeeze(loadmat("data/case_field_2/u.mat")["u"])
    positions = np.squeeze(loadmat("data/case_field_2/position.mat")["p"])
    index = np.random.randint(0, 2583, N_u)  # 4960 triangle
    u_star = solution[index].reshape(-1, 1) + np.random.randn(N_u, 1) * noise_u
    x_star = positions[0, :][index]
    y_star = positions[1, :][index]
    X_star = np.vstack([x_star, y_star]).T

    # 网格配点
    nx = 50
    ny = 50
    xx = np.linspace(0, 1, nx)
    yy = np.linspace(0, 1, ny)
    xxx, yyy = np.meshgrid(xx, yy)
    X_f = np.hstack([xxx.flatten()[:, None], yyy.flatten()[:, None]])
    X_f_, index_hole = collocation_check(X_f)
    collocations = np.vstack([X_f_, X_star])

    # fig = plt.figure()
    # plt.scatter(collocations[:, 0], collocations[:, 1], s=5)
    # plt.savefig("peidian.png")

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
    device = Opt.device
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
    epochs = Opt.epochs
    layers_s = Opt.layers_s
    layers_u = Opt.layers_u

    Net_s = BNNNet(layers_s, st=True).to(device)
    Net_u = BPINN(layers_u, X_star_T, u_star_T, collocations_T, Net_s, in_test=X_f_T_total, out_test=source_field_total,
                  x_bc=bc_f_x_T, y_bc=bc_f_y_T)
    Net_u.train(epochs)
    end_time = time.time()
    print("耗时：", end_time - start_time)

    #  画图
    plt.style.use("default")
    plt.rcParams['font.family'] = 'Times New Roman'
    save = True
    experiment_tag = "_B-PINN_"
    # 预测热源图
    pred_list_plot = Net_u.predict(X_f_T_total)
    mu_plot = np.mean(pred_list_plot, axis=0)
    mu = np.delete(mu_plot, index_hole, axis=0)
    std_plot = np.std(pred_list_plot, axis=0)
    std = np.delete(std_plot, index_hole, axis=0)
    l2_error = np.linalg.norm(source_field_delete.reshape(-1, 1) - mu, 2) / np.linalg.norm(
        source_field_delete, 2)
    r2 = r2_score(source_field_delete, mu)
    print("Ensemble r2:", r2, "---l2_error:", l2_error)

    mask = np.zeros_like(mu_plot, dtype=bool)
    mask[index_hole] = True
    mu_masked = np.ma.array(mu_plot.reshape(nx, ny), mask=mask.reshape(nx, ny))
    std_masked = np.ma.array(std_plot.reshape(nx, ny), mask=mask.reshape(nx, ny))

    fig, ax = plt.subplots(figsize=[8, 3], ncols=2)
    ctf0 = ax[0].contourf(xxx, yyy, mu_masked, cmap="hot", corner_mask=False, levels=100)
    #ax.contour(xxx, yyy, source_field_masked, colors="k", linestyles="dash", corner_mask=False)
    ax[0].set_xlim([0, 1])
    ax[0].set_ylim([0, 1])
    cb0 = plt.colorbar(ctf0, ax=ax[0])
    ctf1 = ax[1].contourf(xxx, yyy, std_masked, cmap="hot", corner_mask=False, levels=100)
    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([0, 1])
    cb1 = plt.colorbar(ctf1)
    if save:
        plt.savefig("training_results/pred_source" + experiment_tag + ".png", bbox_inches="tight", dpi=300)

    # loss
    loss_list = Net_u.loss_list
    plt.style.use('seaborn')
    plt.rcParams['font.family'] = 'Times New Roman'
    fig2 = plt.figure(figsize=[4, 3])
    plt.semilogy()
    plt.plot(loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout()
    if save:
        plt.savefig("training_results/loss" + experiment_tag + ".png", dpi=300, bbox_inches="tight")

    # 对角线
    diag_pred_list_0 = Net_u.predict(xy_diag_0_T)
    diag_pred_list_1 = Net_u.predict(xy_diag_1_T)
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

    fig3 = plt.figure(figsize=[4, 3])
    plt.fill_between(x_diag_0, lower_0.flatten(), upper_0.flatten(), alpha=0.5, rasterized=True,
                     label="Epistemic uncertainty", color="green")
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
    if save:
        plt.savefig("training_results/diag" + experiment_tag + ".png", bbox_inches="tight", dpi=300)

    # 保存数据
    np.savez("training_results/" + experiment_tag + ".npz", total=pred_list_plot, mean=mu_plot, std=std_plot,
             all=pred_list_plot, diag_0=diag_pred_list_0, diag_1=diag_pred_list_1)
