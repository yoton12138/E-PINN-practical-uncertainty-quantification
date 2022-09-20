import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


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


# ==================== 数据准备 ======================
# solution_gp = loadmat("Matlab file/solution_gp.mat")["sol"][101:, :]
# solution_sin = loadmat("Matlab file/solution_sin.mat")["sol"][101:, :]
# gp_data = np.squeeze(loadmat("Matlab file/gp_points.mat")["rf"])
# computation_data = np.load("logvx_DE_Adv_noConst_m5_0.01.npz") #Data/gp/ppt26页/
#
# x_sol = np.linspace(0, 1, 49)
# t_sol = np.linspace(0.01, 0.03, 200)
# X_sol, T_sol = np.meshgrid(x_sol, t_sol)
#
# x_test = np.linspace(0, 1, 409)
# v_test_sin = np.exp(0.5*np.sin(2*np.pi*x_test)) + 0.1  # sin形式
# v_test_gp = np.array([gp_(x, gp_data) for x in x_test])  # gp形式
# m_sin = np.log(v_test_sin - 0.1)
# m_gp = np.log(v_test_gp - 0.1)
#
# # R2 l2 重新计算
# logv_list = computation_data["total"]
# r2_list = computation_data["r2_list"]
# mu = computation_data["mean"]
# std = computation_data["std"]
# lower, upper = (mu - std * 2), (mu + std * 2)
# r2 = round(r2_score(m_gp, mu), 4)
# l2_error = round(np.linalg.norm(np.squeeze(mu) - m_gp, 2) / np.linalg.norm(m_gp, 2), 4)
# print("Ensemble R2: ", r2, "--r_l2: ", l2_error)


# ========================== 绘图 ============================
# 材料形式
plt.style.use("seaborn")
plt.rcParams['font.family'] = 'Times New Roman'
# fig, ax = plt.subplots(figsize=[4, 3])
# ax.plot(x_test, m_sin, linestyle="-", linewidth=2)
# plt.xlabel("x")
# plt.ylabel("log[v(x)-0.1]")
# plt.tight_layout()
# plt.show()
# #plt.savefig("sin.png", bbox_inches="tight", dpi=300)
#
# fig, ax = plt.subplots(figsize=[4, 3])
# ax.plot(x_test, m_gp, linestyle="-", linewidth=2)
# plt.xlabel("x")
# plt.ylabel("log[v(x)-0.1]")
# plt.tight_layout()
# plt.show()
# #plt.savefig("gp.png", bbox_inches="tight", dpi=300)
#
# # 扩散场
# fig, ax = plt.subplots(figsize=[4, 3])
# ctf = ax.contourf(X_sol, T_sol, solution_gp, cmap="Blues", levels=10)
# cb = plt.colorbar(ctf)
# plt.xlabel("x")
# plt.ylabel("t")
# plt.tight_layout()
# plt.show()
# plt.savefig("gp_feild.png", bbox_inches="tight", dpi=300)
#
# fig, ax = plt.subplots(figsize=[4, 3])
# ctf = ax.contourf(X_sol, T_sol, solution_sin, cmap="Blues", levels=10)
# cb = plt.colorbar(ctf)
# plt.xlabel("x")
# plt.ylabel("t")
# plt.tight_layout()
# plt.show()
# plt.savefig("sin_field.png", bbox_inches="tight", dpi=300)

# 画反演结果图
# plt.style.use("seaborn")
# plt.rcParams['font.family'] = 'Times New Roman'
# fig1 = plt.figure(figsize=[4, 3])
# plt.fill_between(x_test, lower.flatten(), upper.flatten(), alpha=0.5, rasterized=True, label="Epistemic uncertainty")
# plt.plot(x_test, m_gp, color="xkcd:orange", linestyle="-", linewidth=2, label="Truth")
# plt.plot(x_test, mu, color="xkcd:dark blue", linestyle="--", linewidth=2, label='Identification')
# plt.legend(loc=3)
# plt.xlabel("x")
# plt.ylabel("log[v(x)-0.1]")
# # plt.ylim([-1.25, 0.75])
# plt.tight_layout()
# plt.show()
# plt.savefig("field.png", bbox_inches="tight", dpi=300)



# 主动采样 gp
# plt.style.use("seaborn-whitegrid")
# plt.rcParams['font.family'] = 'Times New Roman'
# x_lin = np.linspace(0, 8, 9)
# active_r2_list = [0.9336, 0.9896, 0.9916, 0.9971, 0.9976, 0.9984, 0.9984, 0.9995, 0.9995]
# active_l2_list = [0.2576, 0.1018, 0.0915, 0.0541, 0.0495, 0.0405, 0.0396, 0.0224, 0.0218]
# fig4 = plt.figure(figsize=[4, 3])
# plt.plot(x_lin, active_r2_list, linestyle="--", linewidth=2, marker="o",
#          label='Identification')
# plt.xlabel("active sample numbers")
# plt.ylabel("R-square")
# plt.tight_layout()
# plt.show()
# plt.savefig("active_r2.png", bbox_inches="tight", dpi=300, transparent=True)
#
# fig5 = plt.figure(figsize=[4, 3])
# plt.plot(x_lin, active_l2_list, linestyle="--", linewidth=2, marker="o",
#          label='Identification')
# plt.xlabel("active sample numbers")
# plt.ylabel("relative l2 error")
# plt.tight_layout()
# plt.show()
# plt.savefig("active_l2.png", bbox_inches="tight", dpi=300, transparent=True)

# 主动采样 sin
plt.style.use("seaborn-whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'
x_lin = np.linspace(0, 10, 11)
active_r2_list = [0.9911, 0.9920, 0.9961, 0.9963, 0.9965, 0.9989, 0.9990, 0.9991, 0.9993, 0.9997, 0.9996]
active_l2_list = [0.0945, 0.0894, 0.0625, 0.0609, 0.0589, 0.0335, 0.0317, 0.0306, 0.0262, 0.0176, 0.0188]
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