import numpy as np
import math as m
import matplotlib.pyplot as plt

from config import poisson_solver_defaults, C, X, Y
from PoissonSolver import PoissonProblem

import evtk
from pyevtk.hl import pointsToVTK


class fd_solver:
    def __init__(self, ncells_x, ncells_y, T, nt, Re, Fr, Sc, C=C, X=X, Y=Y):

        self.T = T
        self.nt = nt

        self.tau = self.T / self.nt

        self.Re = Re
        self.Fr = Fr
        self.Sc = Sc
        self.C = C

        x = np.linspace(0.0, X, ncells_x + 1, endpoint=True)
        y = np.linspace(-Y, Y, ncells_y + 1, endpoint=True)

        self.hx = x[1] - x[0]
        self.hy = y[1] - y[0]

        xx, yy = np.meshgrid(x, y, indexing="ij")

        self.xx_grid = xx
        self.yy_grid = yy

        self.xx_cell = xx[:-1, :-1] + 0.5 * self.hx
        self.yy_cell = yy[:-1, :-1] + 0.5 * self.hy

        self.xx_u = xx[:, :-1]
        self.yy_u = yy[:, :-1] + 0.5 * self.hy

        self.xx_v = xx[:-1, :] + 0.5 * self.hx
        self.yy_v = yy[:-1, :]

    def DU3(self, u):
        tau = self.tau
        hx = self.hx
        hy = self.hy

        u_p2 = np.pad(u, ((0, 2), (0, 0)), mode="constant")[2:]
        u_p1 = np.pad(u, ((0, 1), (0, 0)), mode="constant")[1:]
        u_n1 = np.pad(u, ((1, 0), (0, 0)), mode="constant")[:-1]

        u_avg = (u_p1 + u) / 2
        c_u = np.abs(u_avg) * tau / hx

        sign_value = u_avg * (u_p1 - u) * 0.5 * (u_p2 - u_p1 - u + u_n1)
        msk = (sign_value >= 0).astype(np.float32)

        switch_lh = (u_avg >= 0) * (
            0.5 * (3.0 - c_u) * u - 0.5 * (1.0 - c_u) * u_n1
        ) + (u_avg < 0) * (0.5 * (3.0 - c_u) * u_p1 - 0.5 * (1.0 - c_u) * u_p2)

        switch_rh = (
            0.5 * (1.0 - tau / hx * u_avg) * u_p1 + 0.5 * (1.0 + tau / hx * u_avg) * u
        )

        return u_avg * (switch_lh * msk + switch_rh * (1.0 - msk))

    def DU1(self, u):
        tau = self.tau
        hx = self.hx
        hy = self.hy

        u_p1 = np.pad(u, ((0, 1), (0, 0)), mode="constant")[1:]
        u_n1 = np.pad(u, ((1, 0), (0, 0)), mode="constant")[:-1]
        u_n2 = np.pad(u, ((2, 0), (0, 0)), mode="constant")[:-2]

        u_avg = (u + u_n1) / 2
        c_u = np.abs(u_avg) * tau / hx

        sign_value = u_avg * (u - u_n1) * 0.5 * (u_n2 - u_n1 - u + u_p1)
        msk = (sign_value >= 0).astype(np.float32)

        switch_lh = (u_avg >= 0) * (
            0.5 * (3.0 - c_u) * u_n1 - 0.5 * (1.0 - c_u) * u_n2
        ) + (u_avg < 0) * (0.5 * (3.0 - c_u) * u - 0.5 * (1.0 - c_u) * u_p1)

        switch_rh = (
            0.5 * (1.0 - tau / hx * u_avg) * u + 0.5 * (1.0 + tau / hx * u_avg) * u_n1
        )

        return u_avg * (switch_lh * msk + switch_rh * (1.0 - msk))

    def DU4(self, u, v):
        tau = self.tau
        hx = self.hx
        hy = self.hy

        u_p2 = np.pad(u, ((0, 0), (0, 2)), mode="constant")[:, 2:]
        u_p1 = np.pad(u, ((0, 0), (0, 1)), mode="constant")[:, 1:]
        u_n1 = np.pad(u, ((0, 0), (1, 0)), mode="constant")[:, :-1]

        v_ = v.copy()
        v_ = np.pad(v_, ((1, 0), (0, 0)), mode="edge")
        v_ = np.pad(v_, ((0, 1), (0, 0)), mode="constant")

        v_avg = (v_[1:] + v_[:-1])[:, 1:] / 2.0

        c_v = np.abs(v_avg) * tau / hy

        sign_value = v_avg * (u_p1 - u) * 0.5 * (u_p2 - u_p1 - u + u_n1)
        msk = (sign_value >= 0).astype(np.float32)

        switch_lh = (v_avg >= 0) * (0.5 * (3 - c_v) * u - 0.5 * (1 - c_v) * u_n1) + (
            v_avg < 0
        ) * (0.5 * (3 - c_v) * u_p1 - 0.5 * (1 - c_v) * u_p2)

        switch_rh = (
            0.5 * (1.0 - tau / hx * v_avg) * u_p1 + 0.5 * (1.0 + tau / hx * v_avg) * u
        )

        return v_avg * (switch_lh * msk + switch_rh * (1.0 - msk))

    def DU2(self, u, v):

        tau = self.tau
        hx = self.hx
        hy = self.hy

        u_p1 = np.pad(u, ((0, 0), (0, 1)), mode="constant")[:, 1:]
        u_n1 = np.pad(u, ((0, 0), (1, 0)), mode="constant")[:, :-1]
        u_n2 = np.pad(u, ((0, 0), (2, 0)), mode="constant")[:, :-2]

        v_ = v.copy()
        v_ = np.pad(v_, ((1, 0), (0, 0)), mode="edge")
        v_ = np.pad(v_, ((0, 1), (0, 0)), mode="constant")

        v_avg = (v_[1:] + v_[:-1])[:, :-1] / 2.0

        c_v = np.abs(v_avg) * tau / hy

        sign_value = v_avg * (u_p1 - u) * 0.5 * (u_p1 - u - u_n1 + u_n2)
        msk = (sign_value >= 0).astype(np.float32)

        switch_lh = (v_avg >= 0) * (
            0.5 * (3.0 - c_v) * u_n1 - 0.5 * (1.0 - c_v) * u_n2
        ) + (v_avg < 0) * (0.5 * (3.0 - c_v) * u - 0.5 * (1.0 - c_v) * u_p1)

        switch_rh = (
            0.5 * (1.0 - tau / hx * v_avg) * u + 0.5 * (1.0 + tau / hx * v_avg) * u_n1
        )

        return v_avg * (switch_lh * msk + switch_rh * (1.0 - msk))

    def DV3(self, u, v):

        tau = self.tau
        hx = self.hx
        hy = self.hy

        u_ = u.copy()
        u_ = np.pad(u_, ((0, 0), (1, 1)), mode="constant")
        u_avg = (u_[:, 1:] + u_[:, :-1])[1:] / 2

        c_u = np.abs(u_avg) * tau / hx

        v_p2 = np.pad(v, ((0, 2), (0, 0)), mode="constant")[2:]
        v_p1 = np.pad(v, ((0, 1), (0, 0)), mode="constant")[1:]
        v_n1 = np.pad(v, ((1, 0), (0, 0)), mode="edge")[:-1]

        sign_value = u_avg * (v_p1 - v) * 0.5 * (v_p2 - v_p1 - v + v_n1)
        msk = (sign_value >= 0).astype(np.float32)

        switch_lh = (u_avg >= 0) * (
            0.5 * (3.0 - c_u) * v - 0.5 * (1.0 - c_u) * v_n1
        ) + (u_avg < 0) * (0.5 * (3.0 - c_u) * v_p1 - 0.5 * (1.0 - c_u) * v_p2)

        switch_rh = (
            0.5 * (1.0 - tau / hx * u_avg) * v_p1 + 0.5 * (1.0 + tau / hx * u_avg) * v
        )

        return u_avg * (switch_lh * msk + switch_rh * (1.0 - msk))

    def DV1(self, u, v):

        tau = self.tau
        hx = self.hx
        hy = self.hy

        u_ = u.copy()
        u_ = np.pad(u_, ((0, 0), (1, 1)), mode="constant")
        u_avg = (u_[:, 1:] + u_[:, :-1])[:-1] / 2
        c_u = np.abs(u_avg) * tau / hx

        v_p1 = np.pad(v, ((0, 1), (0, 0)), mode="constant")[1:]
        v_n1 = np.pad(v, ((1, 0), (0, 0)), mode="edge")[:-1]
        v_n2 = np.pad(v, ((2, 0), (0, 0)), mode="edge")[:-2]

        sign_value = u_avg * (v - v_n1) * 0.5 * (v_p1 - v - v_n1 + v_n2)
        msk = (sign_value >= 0).astype(np.float32)

        switch_lh = (u_avg >= 0) * (0.5 * (3 - c_u) * v_n1 - 0.5 * (1 - c_u) * v_n2) + (
            u_avg < 0
        ) * (0.5 * (3 - c_u) * v - 0.5 * (1 - c_u) * v_p1)

        switch_rh = (
            0.5 * (1.0 - tau / hx * u_avg) * v + 0.5 * (1.0 + tau / hx * u_avg) * v_n1
        )

        return u_avg * (switch_lh * msk + switch_rh * (1.0 - msk))

    def DV4(self, v):

        tau = self.tau
        hx = self.hx
        hy = self.hy

        v_p2 = np.pad(v, ((0, 0), (0, 2)), mode="constant")[:, 2:]
        v_p1 = np.pad(v, ((0, 0), (0, 1)), mode="constant")[:, 1:]
        v_n1 = np.pad(v, ((0, 0), (1, 0)), mode="constant")[:, :-1]

        v_avg = (v_p1 + v) / 2
        c_v = np.abs(v_avg) * tau / hx

        sign_value = v_avg * (v_p1 - v) * 0.5 * (v_p2 - v_p1 - v + v_n1)
        msk = (sign_value >= 0).astype(np.float32)

        switch_lh = (v_avg >= 0) * (
            0.5 * (3.0 - c_v) * v - 0.5 * (1.0 - c_v) * v_n1
        ) + (v_avg < 0) * (0.5 * (3.0 - c_v) * v_p1 - 0.5 * (1.0 - c_v) * v_p2)

        switch_rh = (
            0.5 * (1.0 - tau / hx * v_avg) * v_p1 + 0.5 * (1.0 + tau / hx * v_avg) * v
        )

        return v_avg * (switch_lh * msk + switch_rh * (1.0 - msk))

    def DV2(self, v):

        tau = self.tau
        hx = self.hx
        hy = self.hy

        v_p1 = np.pad(v, ((0, 0), (0, 1)), mode="constant")[:, 1:]
        v_n1 = np.pad(v, ((0, 0), (1, 0)), mode="constant")[:, :-1]
        v_n2 = np.pad(v, ((0, 0), (2, 0)), mode="constant")[:, :-2]

        v_avg = (v + v_n1) / 2
        c_v = np.abs(v_avg) * tau / hx

        sign_value = v_avg * (v - v_n1) * 0.5 * (v_p1 - v - v_n1 + v_n2)
        msk = (sign_value >= 0).astype(np.float32)

        switch_lh = (v_avg >= 0) * (
            0.5 * (3.0 - c_v) * v_n1 - 0.5 * (1 - c_v) * v_n2
        ) + (v_avg < 0) * (0.5 * (3 - c_v) * v - 0.5 * (1.0 - c_v) * v_p1)

        switch_rh = (
            0.5 * (1.0 - tau / hx * v_avg) * v + 0.5 * (1.0 + tau / hx * v_avg) * v_n1
        )

        return v_avg * (switch_lh * msk + switch_rh * (1.0 - msk))

    def DS3(self, u, s):

        tau = self.tau
        hx = self.hx
        hy = self.hy

        s_p2 = np.pad(s, ((0, 2), (0, 0)), mode="constant")[2:]
        s_p1 = np.pad(s, ((0, 1), (0, 0)), mode="constant")[1:]
        s_n1 = np.pad(s, ((1, 0), (0, 0)), mode="edge")[:-1]

        u_avg = u[1:]

        c_u = np.abs(u_avg) * tau / hx

        sign_value = u_avg * (s_p1 - s) * 0.5 * (s_p2 - s_p1 - s + s_n1)
        msk = (sign_value >= 0).astype(np.float32)

        switch_lh = (u_avg >= 0) * (
            0.5 * (3.0 - c_u) * s - 0.5 * (1.0 - c_u) * s_n1
        ) + (u_avg < 0) * (0.5 * (3.0 - c_u) * s_p1 - 0.5 * (1.0 - c_u) * s_p2)

        switch_rh = (
            0.5 * (1.0 - tau / hx * u_avg) * s_p1 + 0.5 * (1.0 + tau / hx * u_avg) * s
        )

        return u_avg * (switch_lh * msk + switch_rh * (1 - msk))

    def DS1(self, u, s):

        tau = self.tau
        hx = self.hx
        hy = self.hy

        s_p1 = np.pad(s, ((0, 1), (0, 0)), mode="constant")[1:]
        s_n1 = np.pad(s, ((1, 0), (0, 0)), mode="edge")[:-1]
        # TODO: check
        s_n2 = np.pad(s, ((2, 0), (0, 0)), mode="edge")[:-2]

        u_avg = u[:-1]
        c_u = np.abs(u_avg) * tau / hx

        sign_value = u_avg * (s - s_n1) * 0.5 * (s_p1 - s - s_n1 + s_n2)
        msk = (sign_value >= 0).astype(np.float32)

        switch_lh = (u_avg >= 0) * (
            0.5 * (3.0 - c_u) * s_n1 - 0.5 * (1.0 - c_u) * s_n2
        ) + (u_avg < 0) * (0.5 * (3.0 - c_u) * s - 0.5 * (1.0 - c_u) * s_p1)

        switch_rh = (
            0.5 * (1.0 - tau / hx * u_avg) * s + 0.5 * (1.0 + tau / hx * u_avg) * s_n1
        )

        return u_avg * (switch_lh * msk + switch_rh * (1 - msk))

    def DS4(self, v, s):

        tau = self.tau
        hx = self.hx
        hy = self.hy

        s_p2 = np.pad(s, ((0, 0), (0, 2)), mode="constant")[:, 2:]
        s_p1 = np.pad(s, ((0, 0), (0, 1)), mode="constant")[:, 1:]
        s_n1 = np.pad(s, ((0, 0), (1, 0)), mode="constant")[:, :-1]

        v_avg = v[:, 1:]
        c_v = np.abs(v_avg) * tau / hx

        sign_value = v_avg * (s_p1 - s) * 0.5 * (s_p2 - s_p1 - s + s_n1)
        msk = (sign_value >= 0).astype(np.float32)

        switch_lh = (v_avg >= 0) * (
            0.5 * (3.0 - c_v) * s - 0.5 * (1.0 - c_v) * s_n1
        ) + (v_avg < 0) * (0.5 * (3.0 - c_v) * s_p1 - 0.5 * (1.0 - c_v) * s_p2)

        switch_rh = (
            0.5 * (1.0 - tau / hx * v_avg) * s_p1 + 0.5 * (1.0 + tau / hx * v_avg) * s
        )

        return v_avg * (switch_lh * msk + switch_rh * (1 - msk))

    def DS2(self, v, s):

        tau = self.tau
        hx = self.hx
        hy = self.hy

        s_p1 = np.pad(s, ((0, 0), (0, 1)), mode="constant")[:, 1:]
        s_n1 = np.pad(s, ((0, 0), (1, 0)), mode="constant")[:, :-1]
        s_n2 = np.pad(s, ((0, 0), (2, 0)), mode="constant")[:, :-2]

        v_avg = v[:, :-1]
        c_v = np.abs(v_avg) * tau / hx

        sign_value = v_avg * (s - s_n1) * 0.5 * (s_p1 - s - s_n1 + s_n2)
        msk = (sign_value >= 0).astype(np.float32)

        switch_lh = (v_avg >= 0) * (
            0.5 * (3.0 - c_v) * s_n1 - 0.5 * (1.0 - c_v) * s_n2
        ) + (v_avg < 0) * (0.5 * (3.0 - c_v) * s - 0.5 * (1.0 - c_v) * s_p1)

        switch_rh = (
            0.5 * (1.0 - tau / hx * v_avg) * s + 0.5 * (1.0 + tau / hx * v_avg) * s_n1
        )

        return v_avg * (switch_lh * msk + switch_rh * (1 - msk))

    def u_tilde(self, u_old, v_old):

        hx = self.hx
        hy = self.hy
        tau = self.tau
        Re = self.Re
        Fr = self.Fr

        u = np.copy(u_old)
        v = np.copy(v_old)

        DU3 = self.DU3(u)
        DU1 = self.DU1(u)
        DU4 = self.DU4(u, v)
        DU2 = self.DU2(u, v)

        v = np.pad(v, ((1, 0), (0, 0)), mode="edge")
        v = np.pad(v, ((0, 1), (0, 0)), mode="constant")
        u = np.pad(u, ((0, 0), (1, 1)), mode="constant")

        u_ = u_old
        u_ -= tau / hx * (DU3 - DU1)
        u_ -= tau / hy * (DU4 - DU2)

        u_ -= (
            tau
            / hy
            / Re
            * (
                ((v[1:, 1:] - v[:-1, 1:]) / hx - (u[:, 2:] - u[:, 1:-1]) / hy)
                - ((v[1:, :-1] - v[:-1, :-1]) / hx - (u[:, 1:-1] - u[:, :-2]) / hy)
            )
        )

        return u_

    def v_tilde(self, u_old, v_old, s_old):

        hx = self.hx
        hy = self.hy
        tau = self.tau
        Re = self.Re
        Fr = self.Fr

        u = np.copy(u_old)
        v = np.copy(v_old)

        DV3 = self.DV3(u, v)
        DV1 = self.DV1(u, v)
        DV4 = self.DV4(v)
        DV2 = self.DV2(v)

        u = np.pad(u, ((0, 0), (1, 1)), mode="constant")
        v = np.pad(v, ((0, 1), (0, 0)), mode="constant")
        v = np.pad(v, ((1, 0), (0, 0)), mode="edge")

        v_ = np.copy(v_old)
        v_ -= tau / hy * (DV4 - DV2)
        v_ -= tau / hx * (DV3 - DV1)

        v_ += (
            tau
            / hx
            / Re
            * (
                ((v[2:] - v[1:-1]) / hx - (u[1:, 1:] - u[1:, :-1]) / hy)
                - ((v[1:-1] - v[:-2]) / hx - (u[:-1, 1:] - u[:-1, :-1]) / hy)
            )
        )

        # TODO: check if correct
        v_ -= tau / Fr * np.pad(s_old, ((0, 0), (1, 0)), mode="constant")

        return v_

    def s_tilde(self, s_old, u_old, v_old):

        hx = self.hx
        hy = self.hy
        tau = self.tau
        Re = self.Re
        Fr = self.Fr
        Sc = self.Sc
        C = self.C

        s = np.copy(s_old)
        u = np.copy(u_old)
        v = np.copy(v_old)

        s_p1_x = np.pad(s, ((0, 1), (0, 0)), mode="constant")[1:]
        s_n1_x = np.pad(s, ((1, 0), (0, 0)), mode="edge")[:-1]
        s_p1_y = np.pad(s, ((0, 0), (0, 1)), mode="constant")[:, 1:]
        s_n1_y = np.pad(s, ((0, 0), (1, 0)), mode="constant")[:, :-1]

        s_ = (
            s
            - tau / hx * (self.DS3(u_old, s) - self.DS1(u_old, s))
            - tau / hy * (self.DS4(v_old, s) - self.DS2(v_old, s))
            + tau
            / (Sc * Re)
            * (
                (s_p1_x - 2.0 * s + s_n1_x) / hx ** 2
                + (s_p1_y - 2.0 * s + s_n1_y) / hy ** 2
            )
            + tau * (v_old[:, 1:] + v_old[:, :-1]) / 2.0 / C
        )

        return s_

    def div_tilde(self, u_tilde, v_tilde):

        hx = self.hx
        hy = self.hy
        tau = self.tau

        return (u_tilde[1:] - u_tilde[:-1]) / hx + (
            v_tilde[:, 1:] - v_tilde[:, :-1]
        ) / hy

    def step(self, u_old, v_old, s_old, p_old):

        hx = self.hx
        hy = self.hy
        tau = self.tau

        u_ = self.u_tilde(u_old, v_old)
        v_ = self.v_tilde(u_old, v_old, s_old)

        rhs = self.div_tilde(u_, v_) / tau

        p = self.solve_poisson(p_old, rhs)
        p -= np.mean(p)

        p_ = np.pad(p, ((1, 1), (1, 1)), mode="edge")

        u_next = u_ - tau / hx * (p_[1:, 1:-1] - p_[:-1, 1:-1])
        v_next = v_ - tau / hy * (p_[1:-1:, 1:] - p_[1:-1:, :-1])

        s_next = self.s_tilde(s_old, u_next, v_next)

        return u_next, v_next, s_next, p

    def solve_poisson(self, p0, rhs):

        p = PoissonProblem(p_init=p0, rhs=rhs, h_x=self.hx, h_y=self.hy)
        p.solve(**poisson_solver_defaults)

        return p.get_solution()

    def dump_vtk(self, fname: str, u, v, s, p):
        s_p1 = np.pad(s, [(0, 1), (0, 0)], mode='constant')[1:]

        s_deriv = (s_p1 - s)/self.hx

        return pointsToVTK(
            fname + "_cell_data",
            self.xx_cell.flatten(),
            self.yy_cell.flatten(),
            np.zeros_like(self.xx_cell.flatten()),
            data={
                "salinity": s.flatten(),
                "salinity_dx": s_deriv.flatten(),
                "pressure": p.flatten(),
            },
        )
