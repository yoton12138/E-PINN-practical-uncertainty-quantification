import matplotlib.pyplot as plt
import numpy as np
from utils import *


# ======================= 准备数据 =========================
npz_data = np.load("1.npz")
s_pred_list = npz_data["total"]
# s_pred_list_plot = npz_data["all"]
mu = npz_data["mean"]
std = npz_data["std"]
diag_pred_list_0 = npz_data["diag_0"]  # 有的数据没有这两个对角
diag_pred_list_1 = npz_data["diag_1"]

# 基本绘图元素
# 网格配点
nx = 50
ny = 50
xx = np.linspace(0, 1, nx)
yy = np.linspace(0, 1, ny)
xxx, yyy = np.meshgrid(xx, yy)
X_f = np.hstack([xxx.flatten()[:, None], yyy.flatten()[:, None]])
index_hole = np.where((X_f[:, 0] < 0.7) & (X_f[:, 0] > 0.5) & (X_f[:, 1] < 0.7) & (X_f[:, 1] > 0.5))[0]
X_f_ = np.delete(X_f, index_hole, axis=0)

# 对角测试
x_diag_0 = np.linspace(0, 0.5, 101)
x_diag_1 = np.linspace(0.7, 1.0, 101)
xy_diag_0 = np.vstack([x_diag_0, x_diag_0]).T
xy_diag_1 = np.vstack([x_diag_1, x_diag_1]).T

# ======================= 画图 =========================
# 画默认白底风格的图
plt.style.use("default")
plt.rcParams['font.family'] = 'Times New Roman'
# 画均值热源图
mask = np.zeros_like(mu, dtype=bool)
mask[index_hole] = True
mu_masked = np.ma.array(mu.reshape(nx, ny), mask=mask.reshape(nx, ny))
std_masked = np.ma.array(std.reshape(nx, ny), mask=mask.reshape(nx, ny))

fig, ax = plt.subplots(figsize=[4, 3])
ctf = ax.contourf(xxx, yyy, mu_masked, cmap="hot", corner_mask=False, levels=np.linspace(0, 2, 101))  # 设置level最管用
# ax.contour(xxx, yyy, source_field_masked, colors="k", linestyles="dash", corner_mask=False)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
cb = plt.colorbar(ctf)
plt.show()
plt.savefig("inversion_mu.png", bbox_inches="tight", dpi=300)

fig, ax = plt.subplots(figsize=[4, 3])
ctf = ax.contourf(xxx, yyy, std_masked, cmap="hot", corner_mask=False, levels=101)  # 设置level最管用
# ax.contour(xxx, yyy, source_field_masked, colors="k", linestyles="dash", corner_mask=False)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.plot(x_diag_0, x_diag_0, linestyle="--", color="white")
ax.plot(x_diag_1, x_diag_1, linestyle="--", color="white")
cb = plt.colorbar(ctf)
plt.show()
plt.savefig("inversion_std.png", bbox_inches="tight", dpi=300)


plt.style.use('seaborn')
plt.rcParams['font.family'] = 'Times New Roman'
# 对角线图 左下右上
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
plt.show()

