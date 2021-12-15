import numpy as np
from config import *


def build_default_initial_conditions(nx, ny):

    x_scale = R_0
    y_scale = R_0
    t_scale = 1.0 / N
    u_scale = R_0 * N
    rho_scale = rho_0
    p_scale = rho_0 * R_0 ** 2 * N ** 2

    x_space = np.linspace(0.0, X, nx)
    y_space = np.linspace(-Y, Y, ny)
    x_space /= x_scale
    y_space /= y_scale

    xx, yy = np.meshgrid(x_space, y_space)

    # initial velocity field is zero-valued
    u_0_distr = np.zeros_like(xx)
    v_0_distr = np.zeros_like(xx)

    # define area where the spot is located
    spot_mask = (
        (xx - spot_center_x / x_scale) ** 2 + (yy - spot_center_y / y_scale) ** 2
    ) <= R_0

    # initial soft disturbance
    s_0_distr = spot_mask * yy / Lambda

    rho_0_distr = rho_0 * (1.0 - yy / Lambda + s_0_distr)
    rho_0_distr = rho_0_distr / rho_scale

    p_0_distr = (-1.0) * rho_0 * g * yy * spot_mask + (-1.0) * rho_0 * g * (
        yy - 1 / Lambda / 2 * yy ** 2
    ) * (1.0 - spot_mask)

    p_0_distr = p_0_distr / p_scale

    return (
        x_space[1] - x_space[0],
        y_space[1] - y_space[0],
        u_0_distr,
        v_0_distr,
        p_0_distr,
        s_0_distr,
    )
