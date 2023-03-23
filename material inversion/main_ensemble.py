from networks import *
from options import Opt
from utils import *
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("agg")
seed_torch(1234)

if __name__ == "__main__":
    # 观测数据准备
    measures = loadmat("Matlab file/measures_gp_pure.mat")["measure"]
    solution = loadmat("Matlab file/solution_gp.mat")["sol"][101:, :]
    gp_data = np.squeeze(loadmat("Matlab file/gp_points.mat")["rf"])

    x_sol = np.linspace(0, 1, 49)
    t_sol = np.linspace(0.01, 0.03, 200)
    X_sol, T_sol = np.meshgrid(x_sol, t_sol)
    X_fem = np.hstack([X_sol.flatten()[:, None], T_sol.flatten()[:, None]])
    u_fem = solution.reshape(-1, 1)

    x_obs = np.linspace(0, 1, 13)
    t_obs = np.linspace(0.01, 0.03, 9)
    t_0 = np.zeros_like(x_obs)
    X_0 = np.vstack([x_obs, t_0]).T
    X_obs, T_obs = np.meshgrid(x_obs, t_obs)
    X_star = np.hstack([X_obs.flatten()[:, None], T_obs.flatten()[:, None]])
    # X_star = np.vstack([X_0, X_star])
    u_star = measures.reshape(-1, 1) + np.random.randn(117, 1)*0.01  # noise***********
    # u_star = np.vstack([np.zeros_like(x_obs).reshape(-1, 1), u_star])

    x_test = np.linspace(0, 1, 409)
    # v_test = np.exp(0.5*np.sin(2*np.pi*x_test)) + 0.1  # sin形式
    # v_test = np.array([square_wave(x) for x in x_test])  # 分段方波形式
    v_test = np.array([gp_(x, gp_data) for x in x_test])  # gp形式
    m = np.log(v_test - 0.1)

    X_BC, T_BC = np.meshgrid(np.array([0, 1.0]), t_sol)
    XT_BC = np.hstack([X_BC.flatten()[:, None], T_BC.flatten()[:, None]])

    # 网格配点
    nx = 50
    nt = 100
    xx = np.linspace(0, 1, nx+2)[1:-1]
    tt = np.linspace(0.01, 0.03, nt+2)[1:-1]
    xxx, ttt = np.meshgrid(xx, tt)
    X_f = np.hstack([xxx.flatten()[:, None], ttt.flatten()[:, None]])

    X_f = np.vstack([X_f, X_star])

    # 主动加点
    x_test_const = np.array([0, 0.34, 0.6])
    #v_test_const = np.exp(0.5*np.sin(2*np.pi*x_test_const)) + 0.1  # sin形式
    v_test_const = np.array([gp_(x, gp_data) for x in x_test_const])  # gp形

    # numpy to tensor
    X_star_T = torch.tensor(X_star, dtype=torch.float32, requires_grad=True).to(Opt.device)
    u_star_T = torch.tensor(u_star, dtype=torch.float32).to(Opt.device)
    X_f_T = torch.tensor(X_f, dtype=torch.float32, requires_grad=True).to(Opt.device)
    x_test_T = torch.tensor(x_test, dtype=torch.float32).to(Opt.device)
    X_fem_T = torch.tensor(X_fem, dtype=torch.float32, requires_grad=True).to(Opt.device)
    u_fem_T = torch.tensor(u_fem, dtype=torch.float32).to(Opt.device)
    XT_BC_T = torch.tensor(XT_BC, dtype=torch.float32, requires_grad=True).to(Opt.device)
    # 网络相关
    """
    构造一个神经网络Net_u拟合场数据，一个神经网络Net_v拟合扩散率v
    """
    start_time = time.time()
    epochs = Opt.epochs
    layers_v = Opt.layers_v
    layers_u = Opt.layers_u

    Net_v = Fnn(layers_v, st=False).to(Opt.device)
    Net_u = PINN(layers_u, X_star_T, u_star_T, X_f_T, Net_v, x_test_T, v_test, xt_bc=XT_BC_T)
    Net_u.train(epochs, pre_train=True)
    end_time = time.time()
    print("耗时：", end_time - start_time)

    # 两种预测方式， 直接预测，
    # mu, std = Net_u.predict(x_test_T)

    # 转化log预测
    logv_list = []
    r2_list = []
    for i in range(Opt.M):
        v_pre = Net_u.predict_m(x_test_T, m=int(i))
        logv = np.log(np.asarray(v_pre - 0.1))
        r2_i = r2_score(np.log(v_test - 0.1), logv)
        logv_list.append(logv)
        r2_list.append(r2_i)
    mu = np.mean(logv_list, axis=0)
    std = np.std(logv_list, axis=0)
    lower, upper = (mu - std * 2), (mu + std * 2)
    r2 = r2_score(np.log(v_test - 0.1), mu)
    r2_list.append(r2)
    print("Ensemble mean R2: ", r2)

    loss_list = Net_u.loss_list

    # 自定义可视化标准单图回归不确定性图例模板
    experiment_tag = "_DE_Adv_m5_0.01"  # 便捷更改Pdf图等的名字
    save = True
    # 保存一些数据
    np.savez("training_results/logvx" + experiment_tag + ".npz", total=logv_list, mean=mu, std=std, loss_list=loss_list, r2_list=r2_list)

    plt.style.use("seaborn")
    plt.rcParams['font.family'] = 'Times New Roman'
    fig1 = plt.figure(figsize=[4, 3])
    plt.fill_between(x_test, lower.flatten(), upper.flatten(), alpha=0.5, rasterized=True, label="Epistemic uncertainty")
    plt.plot(x_test, np.log(v_test - 0.1), color="xkcd:orange", linestyle="-", linewidth=2, label="Truth")
    plt.plot(x_test, mu, color="xkcd:dark blue", linestyle="--", linewidth=2, label='Identification')
    plt.legend(loc=3)
    plt.xlabel("x")
    plt.ylabel("log[v(x)-0.1]")
    plt.tight_layout()
    if save:
        plt.savefig("training_results/vx" + experiment_tag + ".png", dpi=600)

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

    fig3 = plt.figure(figsize=[4, 3])
    plt.plot(x_test, np.log(v_test-0.1), color="xkcd:orange", linestyle="-", linewidth=2, label="Truth")
    for i in range(Opt.M):
        v_pre = Net_u.predict_m(x_test_T.view(-1, 1), m=int(i))
        if i == 0:
            plt.plot(x_test, np.log(v_pre-0.1), color="xkcd:dark blue", alpha=0.3, linestyle="-", linewidth=2, label='Identification')
        else:
            plt.plot(x_test, np.log(v_pre-0.1), color="xkcd:dark blue", alpha=0.3, linestyle="-", linewidth=2)
    plt.legend(loc=3)
    plt.xlabel("x")
    plt.ylabel("log[v(x)-0.1]")
    plt.tight_layout()
    if save:
        plt.savefig("training_results/samples" + experiment_tag + ".png", dpi=600)
