import numpy as np
import torch


def seed_torch(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 热源场函数
def source_func(x):
    s1 = 1.0 * np.exp(-0.5 * ((x[:, 0] - 0.3)**2 + (x[:, 1] - 0.4)**2) / 0.15**2)

    return s1


def source_func_2(x):
    s1 = 1.0 * np.exp(-0.5 * ((x[:, 0] - 0.3)**2 + (x[:, 1] - 0.4)**2) / 0.15**2)
    s2 = 2.0 * np.exp(-0.5 * ((x[:, 0] - 0.8)**2 + (x[:, 1] - 0.8)**2) / 0.05**2)

    return s1 + s2


def collocation_check(X_f):
    # 输入配点坐标对， 删除不在几何之内的点
    eps = 1e-5
    index_hole = np.where((X_f[:, 0] < 0.7-eps) & (X_f[:, 0] > 0.5+eps) &
                          (X_f[:, 1] < 0.7-eps) & (X_f[:, 1] > 0.5+eps))[0]
    X_f = np.delete(X_f, index_hole, axis=0)

    index_outside = np.where((X_f[:, 0] < 0.-eps) | (X_f[:, 0] > 1.+eps) |
                            (X_f[:, 1] < 0.-eps) | (X_f[:, 1] > 1.+eps))[0]
    X_f = np.delete(X_f, index_outside, axis=0)

    return X_f, index_hole


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]

        return x

