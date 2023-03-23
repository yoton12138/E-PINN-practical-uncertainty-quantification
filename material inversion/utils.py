import numpy as np
import torch
import os, re
from options import Opt

def compute_covariance_matrix(X, s, ell):
    """
    Computes the covariance matrix at ``X``. This simply computes a
    squared exponential covariance and is here for illustration only.
    We will be using covariances from this package:
    `GPy <https://github.com/SheffieldML/GPy>`_.

    :param X:   The evaluation points. It has to be a 2D numpy array of
                dimensions ``num_points x input_dim``.
    :type X:    :class:`numpy.ndarray``
    :param s:   The signal strength of the field. It must be positive.
    :type s:    float
    :param ell: A list of lengthscales. One for each input dimension. The must
                all be positive.
    :type ell:  :class:`numpy.ndarray`
    """
    assert X.ndim == 2
    assert s > 0
    assert ell.ndim == 1
    assert X.shape[1] == ell.shape[0]
    C = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                dx = (X[i, k] - X[j, k]) / ell[k]
                C[i, j] += s**2 * np.exp(-0.5*dx*dx)
        C[i, i] = C[i, i] + 1e-6

    return C


def Corr(X, theta=1.0, l=0.2):
    nx = X.shape[0]
    C = torch.zeros((nx, nx))
    V1, V2 = torch.meshgrid(torch.squeeze(X), torch.squeeze(X))
    V = torch.cat([V1.flatten()[:, None], V2.flatten()[:, None]], dim=1)
    d = (V[:, 0] - V[:, 1])**2
    C = theta * torch.exp(-0.5 * d / l**2)
    C = C.view(nx, nx)
    eps = (torch.eye(nx) * 1e-6).to(Opt.device)
    C = C + eps

    return C


def square_wave(x):
    if x < 0.4 or x > 0.7:
        return np.exp(0.5) + 0.1
    else:
        return np.exp(-0.5) + 0.1


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


def find_all_files(path):
    for root, ds, fs in os.walk(path):
        for f in fs:
            if re.match(r'.*\d.*', f):
                fullname = os.path.join(root, f)
                yield fullname


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]

        return x


def seed_torch(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True