# 配置参数
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using device: ", torch.cuda.get_device_name(0))


# class Opt(object):
#     """for E-PINN MC-dropout"""
#     device = device
#     noise = 0.01
#     epochs = 50000
#     lr_u = 0.003
#     lr_v = 0.005
#
#     lw_mse = 1.
#     lw_pde = 0.001
#     lw_bc = 0.001
#
#     layers_v = [1, 20, 20, 20, 20, 1]
#     layers_u = [2, 20, 20, 20, 20, 1]
#     #
#     M = 5


class Opt(object):
    """for B-PINN"""
    device = device
    noise = 0.01
    epochs = 50000
    lr_u = 0.003
    lr_v = 0.005

    lw_mse = 1.
    lw_pde = 0.001
    lw_bc = 0.001
    lw_kl = 0.0002

    layers_v = [1, 10, 10, 10, 10, 1]
    layers_u = [2, 20, 20, 20, 20, 1]

