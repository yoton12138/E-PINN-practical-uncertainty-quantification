import matplotlib
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

    Net_s = Fnn(layers_s, st=True).to(device)
    Net_u = PINN(layers_u, X_star_T, u_star_T, collocations_T, Net_s, in_test=X_f_T_total, out_test=source_field_total,
                 x_bc=bc_f_x_T, y_bc=bc_f_y_T)
    Net_u.train(epochs)
    # Net_u.train_batch(epochs)
    end_time = time.time()
    print("耗时：", end_time - start_time)

    # 绘图
    plt.style.use("default")
    plt.rcParams['font.family'] = 'Times New Roman'
    save = True
    experiment_tag = "_DE_m10_eps0.0_noise_0.02"
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
    s_pred_list_plot = []
    s_pred_list = []
    r2_list = []
    for i in range(Net_u.M):
        s_pred_i_plot = Net_u.predict_m(X_f_T_total, m=i)
        s_pred_i = np.delete(s_pred_i_plot, index_hole, axis=0)
        s_pred_list_plot.append(s_pred_i_plot)
        s_pred_list.append(s_pred_i)
        r2_list.append(r2_score(source_field_delete, s_pred_i))
    mu = np.mean(s_pred_list_plot, axis=0)
    std = np.std(s_pred_list_plot, axis=0)
    l2_error = np.linalg.norm(source_field_delete.reshape(-1, 1) - np.mean(s_pred_list, axis=0), 2) / np.linalg.norm(source_field_delete, 2)
    r2 = r2_score(source_field_delete, np.mean(s_pred_list, axis=0))
    print("Ensemble r2:", r2, "---l2_error:", l2_error)

    # mu, std = Net_u.predict(X_f_T_total)
    mask = np.zeros_like(mu, dtype=bool)
    mask[index_hole] = True
    mu_masked = np.ma.array(mu.reshape(nx, ny), mask=mask.reshape(nx, ny))
    std_masked = np.ma.array(std.reshape(nx, ny), mask=mask.reshape(nx, ny))

    fig, ax = plt.subplots(figsize=[8, 3], ncols=2)
    ctf0 = ax[0].contourf(xxx, yyy, mu_masked, cmap="hot", corner_mask=False, levels=100)
    # ax.contour(xxx, yyy, source_field_masked, colors="k", linestyles="dash", corner_mask=False)
    ax[0].set_xlim([0, 1])
    ax[0].set_ylim([0, 1])
    cb0 = plt.colorbar(ctf0, ax=ax[0])
    ctf1 = ax[1].contourf(xxx, yyy, std_masked, cmap="hot", corner_mask=False, levels=100)
    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([0, 1])
    cb1 = plt.colorbar(ctf1)
    plt.show()
    if save:
        plt.savefig("pred_source" + experiment_tag + ".png", bbox_inches="tight", dpi=300)

    # 单一预测查看
    fig, axes = plt.subplots(figsize=[12, 6], ncols=3, nrows=2)
    for i in range(2):
        for j in range(3):
            k = i * 3 + j
            if k < 5:
                pred_i_masked = np.ma.array(s_pred_list_plot[k].reshape(nx, ny), mask=mask.reshape(nx, ny))
                ctf = axes[i, j].contourf(xxx, yyy, pred_i_masked, cmap="hot", corner_mask=False, levels=100)
                axes[i, j].set_xlim([0, 1])
                axes[i, j].set_ylim([0, 1])
                cb = plt.colorbar(ctf, ax=axes[i, j])
            else:
                ctf = axes[i, j].contourf(xxx, yyy, mu_masked, cmap="hot", corner_mask=False, levels=100)
                axes[i, j].set_xlim([0, 1])
                axes[i, j].set_ylim([0, 1])
                cb = plt.colorbar(ctf, ax=axes[i, j])
    plt.show()
    if save:
        plt.savefig("m_pred" + experiment_tag + ".png", bbox_inches="tight", dpi=300)

    # loss
    loss_list = Net_u.loss_list
    plt.style.use('seaborn')
    plt.rcParams['font.family'] = 'Times New Roman'
    fig2 = plt.figure(figsize=[4, 3])
    for i, loss_m in enumerate(loss_list):
        plt.plot(loss_m, label="model_" + str(i), alpha=0.8)
    plt.semilogy()
    plt.legend(loc=1)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("loss" + experiment_tag + ".png", dpi=300, bbox_inches="tight")

    # 对角线
    diag_pred_list_0 = []
    diag_pred_list_1 = []
    for i in range(Net_u.M):
        diag_pred_0_i_plot = Net_u.predict_m(xy_diag_0_T, m=i)
        diag_pred_1_i_plot = Net_u.predict_m(xy_diag_1_T, m=i)
        diag_pred_list_0.append(diag_pred_0_i_plot)
        diag_pred_list_1.append(diag_pred_1_i_plot)
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
    if save:
        plt.savefig("diag" + experiment_tag + ".png", bbox_inches="tight", dpi=300)

    # 保存数据
    np.savez(experiment_tag + ".npz", total=s_pred_list, mean=mu, std=std, all=s_pred_list_plot,
             diag_0=diag_pred_list_0, diag_1=diag_pred_list_1)