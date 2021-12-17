import numpy as np
from config import *
from numerical_scheme import fd_solver

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def build_default_initial_conditions(solver: fd_solver):
    # initial velocity field is zero-valued
    u_0_distr = np.zeros_like(solver.xx_u)
    v_0_distr = np.zeros_like(solver.xx_v)

    # coordinates of cells
    xx = solver.xx_cell
    yy = solver.yy_cell

    # define area where the spot is located
    # size of the spot in dimensionless coordinates is 1
    spot_mask = (xx - spot_center_x) ** 2 + (yy - spot_center_y) ** 2 <= 1.0
    spot_mask = spot_mask.astype(np.int32)

    # initial soft disturbance
    s_0_distr = spot_mask * yy / C
    p_0_distr = C * yy * spot_mask + 0.5 * yy ** 2

    return (
        u_0_distr,
        v_0_distr,
        p_0_distr,
        s_0_distr,
    )


def plot_fields(u, v, s, p):
    # to do: divide by hx
    delta_s_x = s[1:, :] - s[:-1, :]
    data = [u, v, p, s, delta_s_x]
    labels = ["$u$", "$v$", "$p$", "$s$", r"$\frac{\partial s}{\partial x}$"]
    fig, axs = plt.subplots(nrows=3, ncols=2)
    axs = axs.flatten()

    for _idx, datum in enumerate(data):
        im = axs[_idx].matshow(datum.T)

        divider = make_axes_locatable(axs[_idx])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        axs[_idx].set_title(labels[_idx])

    plt.show()
