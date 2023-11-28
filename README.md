# E-PINN-practical-uncertainty-quantification
source code and examples. The arcticle is availiable at [Practical uncertainty quantification for space-dependent inverse heat conduction problem via ensemble physics-informed neural networks](https://www.sciencedirect.com/science/article/pii/S0735193323003299).

equation (12) is not correct, $\mathbb{E}_{\theta(\chi, d) \sim \mathscr{D}}$

should be $\mathbb{E}_{(\chi, d) \sim \mathscr{D}}$

损失函数的符号 L 和 $\mathscr{L}$ 存在混用，这是编辑没有统一，作者在此深表歉意。

# Citation
If this code is relevant for your research, please consider citing:
```
@article{JIANG2023106940,
title = {Practical uncertainty quantification for space-dependent inverse heat conduction problem via ensemble physics-informed neural networks},
author = {Xinchao Jiang and Xin Wang and Ziming Wen and Enying Li and Hu Wang},
journal = {International Communications in Heat and Mass Transfer},
volume = {147},
pages = {106940},
year = {2023},
issn = {0735-1933},
doi = {https://doi.org/10.1016/j.icheatmasstransfer.2023.106940},
url = {https://www.sciencedirect.com/science/article/pii/S0735193323003299},
}
```
