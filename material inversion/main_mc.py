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
    u_star = measures.reshape(-1, 1) + np.random.randn(117, 1)*Opt.noise  # noise***********

    x_test = np.linspace(0, 1, 409)
    # v_test = np.exp(0.5*np.sin(2*np.pi*x_test)) + 0.1  # sin形式
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

    Net_v = Fnn(layers_v, st=False, dropout=True).to(Opt.device)
    model = MCDropout(layers_u, X_star_T, u_star_T, X_f_T, Net_v, x_test_T, v_test, xt_bc=XT_BC_T)
    model.train(epochs, pre_train=True)
    end_time = time.time()
    print("耗时：", end_time - start_time)

    u_fem_pred = model.predict_u(X_fem_T)
    r2_u = r2_score(u_fem, u_fem_pred)
    r_l2_error_u = np.linalg.norm(solution.reshape(-1, 1) - u_fem_pred, 2) / np.linalg.norm(solution.reshape(-1, 1), 2)
    plt.figure(2, figsize=[8, 6])
    plt.contourf(X_sol, T_sol, u_fem_pred.reshape(200, 49))
    plt.savefig(f"training_results/final_u_pred.png", dpi=600)

    plt.style.use("seaborn")
    plt.rcParams['font.family'] = 'Times New Roman'

    loss = model.loss_list
    fig2 = plt.figure(figsize=[4, 3])
    plt.plot(loss, alpha=0.1*7)
    plt.semilogy()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("training_results/loss.png", dpi=600)

    vx = []
    for i in range(100):
        vx_T = Net_v(x_test_T.view(-1, 1))
        vx_i = np.log(vx_T.detach().cpu().numpy() - 0.1)
        vx.append(vx_i)
    vx_mean = np.squeeze(np.mean(vx, axis=0))
    vx_std = np.squeeze(np.std(vx, axis=0))
    lower = vx_mean - 2*vx_std
    upper = vx_mean + 2*vx_std
    r2_v = r2_score(m, vx_mean)
    r_l2_error_v = np.linalg.norm(m - np.squeeze(vx_mean), 2) / np.linalg.norm(m, 2)
    print(f"r2: {r2_v}, l2: {r_l2_error_v} ")

    fig1 = plt.figure(figsize=[4, 3])
    plt.fill_between(x_test, lower, upper, alpha=0.5, rasterized=True, label="Epistemic uncertainty")
    plt.plot(x_test, m, color="xkcd:orange", linestyle="-", linewidth=2, label="Truth")
    plt.plot(x_test, vx_mean, color="xkcd:dark blue", linestyle="--", linewidth=2, label='Identification')
    plt.legend(loc=3)
    plt.xlabel("x")
    plt.ylabel("log[v(x)-0.1]")
    plt.ylim([-1.25, 0.75])  # gp
    plt.tight_layout()
    plt.savefig(f"training_results/final_v_pred.png", dpi=600, bbox_inches="tight")

