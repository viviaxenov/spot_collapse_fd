import numpy as np
import argparse

from numerical_scheme import fd_solver
from config import *
from utils import *


def parse_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--nx", type=int, dest="nx", required=True, help="Number of nodes along X"
    )
    parser.add_argument(
        "--ny", type=int, dest="ny", required=True, help="Number of nodes along Y"
    )
    parser.add_argument(
        "--T", type=float, dest="T", required=True, help="Total time of integration"
    )

    parser.add_argument(
        "--n_steps", type=int, dest="nt", required=True, help="Number of timesteps"
    )

    return parser.parse_args()


def main():

    args = vars(parse_args())

    nx = args["nx"]
    ny = args["ny"]
    nt = args["nt"]
    T = args["T"]

    solver = fd_solver(nx, ny, T, nt, Re, Fr, Sc)

    u_0, v_0, p_0, s_0 = build_default_initial_conditions(solver)
    u, v, s, p = u_0, v_0, s_0, p_0

    print(p.shape, solver.xx_cell.shape)

    for cur_iter in range(solver.nt):
        u, v, s, p = solver.step(u, v, s, p)
        if cur_iter % 10 == 0:
            solver.dump_vtk(f"output/{cur_iter:05d}", u, v, s, p)

    solver.dump_vtk(f"output/{cur_iter:05d}", u, v, s, p)
    plot_fields(u, v, s, p)

    return


if __name__ == "__main__":

    main()
