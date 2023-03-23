# 配置参数
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using device: ", torch.cuda.get_device_name(0))


class Opt(object):
    """ for E-PINN"""
    device = device
    noise = 0.05
    epochs = 50000
    lr = 0.001
    lw_pde = 0.0001
    lw_bc = 1
    layers_u = [3, 20, 20, 20, 20, 1]
    M = 5


# class Opt(object):
#     """ for B-PINN """
#     device = device
#     noise = 0.05
#     epochs = 50000
#     lr = 0.01
#     lw_pde = 1
#     lw_bc = 10
#     lw_kl = 0.02
#     layers_u = [3, 10, 10, 10, 10, 1]
#     M = 5
#
#
# class Opt(object):
#     """ for MC-dropout """
#     device = device
#     noise = 0.05
#     epochs = 50000
#     lr = 0.001
#     lw_pde = 0.001
#     lw_bc = 1
#     layers_u = [3, 20, 20, 20, 20, 1]
#     M = 5