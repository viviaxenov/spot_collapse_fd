import numpy as np
import math as m

from config import poisson_solver_defaults
from PoissonSolver import PoissonProblem

class fd_solver():

        def __init__(
                self, hx, hy, T, nt, 
                Re, Fr, Sc
        ):

                self.hx = hx
                self.hy = hy  
                self.T  = T
                self.nt = nt

                self.n_steps = self.T / float(self.nt) 

                self.Re = Re
                self.Fr = Fr
                self.Sc = Sc

        @staticmethod
        def DU3(u):

                u_p2 = np.pad(u, ((0, 2), (0, 0)), mode='constant')[2:]
                u_p1 = np.pad(u, ((0, 1), (0, 0)), mode='constant')[1:]
                u_n1 = np.pad(u, ((1, 0), (0, 0)), mode='constant')[:-1]

                u_avg = (u_p1 + u) / 2 
                c_u   = np.abs(u_avg) * tau / hx
                sign_value = u_avg * (u_p1 - u) * 0.5 * (u_p2 - u_p1 - u + u_n1)
                msk = sign_value >= 0 

                switch_lh = (u_avg >= 0) * (
                        0.5 * (3 - c_u) * u - 0.5 * (1 - c_u) * u_n1
                ) + (u_avg < 0) * (
                        0.5 * (3 - c_u) * u_p1 - 0.5 * (1 - c_u) * u_p2
                )

                switch_rh = 0.5 * (1. - tau / hx * u_avg) * u_p1 -\
                                        0.5 * (1. + tau / hx * u_avg) * u

                return u_avg * (switch_lh * msk + switch_rh * (1 - msk))

        @staticmethod
        def DU1(u):
                
                u_p1 = np.pad(u, ((0, 1), (0, 0)), mode='constant')[1:]
                u_n1 = np.pad(u, ((1, 0), (0, 0)), mode='constant')[:-1]
                u_n2 = np.pad(u, ((2, 0), (0, 0)), mode='constant')[:-2]

                u_avg = (u + u_n1) / 2
                c_u   = np.abs(u_avg) * tau / hx

                sign_value = u_avg * (u - u_n1) * 0.5 * (u_n2 - u_n1 - u + u_p1)
                msk = sign_value >= 0 
                switch_lh = (u_avg >= 0) * (
                        0.5 * (3 - c_u) * u_n1 - 0.5 * (1 - c_u) * u_n2
                ) + (u_avg < 0) * (
                0.5 * (3 - c_u) * u - 0.5 * (1 - c_u) * u_p1
                )

                switch_rh = 0.5 * (1. - tau / hx * u_avg) * u -\
                                        0.5 * (1. + tau / hx * u_avg) * u_n1

                return u_avg * (switch_lh * msk + switch_rh * (1 - msk))

        @staticmethod
        def DU4(u, v):

                u_p2 = np.pad(u, ((0, 0), (0, 2)), mode='constant')[:, 2:]
                u_p1 = np.pad(u, ((0, 0), (0, 1)), mode='constant')[:, 1:]
                u_n1 = np.pad(u, ((0, 0), (1, 0)), mode='constant')[:, :-1]
                v_n1 = np.pad(v, ((1, 0), (0, 0)), mode='constant')[:-1]
                
                v_avg = (v + v_n1) / 2
                c_v   = np.abs(v_avg) * tau / hy

                sign_value = v_avg * (u_p1 - u) * 0.5 * (u_p2 - u_p1 - u + u_n1)
                msk = sign_value >= 0 

                switch_lh = (v_avg >= 0) * (
                        0.5 * (3 - c_v) * u - 0.5 * (1 - c_v) * u_n1
                ) + (v_avg < 0) * (
                        0.5 * (3 - c_v) * u_p1 - 0.5 * (1 - c_v) * u_p2
                )

                switch_rh = 0.5 * (1. - tau / hx * v_avg) * u_p1 -\
                                        0.5 * (1. + tau / hx * v_avg) * u

                return v_avg * (switch_lh * msk + switch_rh * (1 - msk))

        @staticmethod
        def DU2(u, v):

                u_p1 = np.pad(u, ((0, 0), (0, 1)), mode='constant')[:, 1:]
                u_n1 = np.pad(u, ((0, 0), (1, 0)), mode='constant')[:, :-1]
                u_n2 = np.pad(u, ((0, 0), (2, 0)), mode='constant')[:, :-2]

                v    = np.pad(v, ((0, 0), (1, 0)), mode='constant')[:, :-1]
                v_n1 = np.pad(v, ((1, 0), (0, 0)), mode='constant')[:-1]

                v_avg = (v + v_n1) / 2
                c_v   = np.abs(v_avg) * tau / hy

                sign_value = v_avg * (u_p1 - u) * 0.5 * (u_p1 - u - u_n1 + u_n2)
                msk = sign_value >= 0 

                switch_lh = (v_avg >= 0) * (
                        0.5 * (3 - c_v) * u_n1 - 0.5 * (1 - c_v) * u_n2
                ) + (v_avg < 0) * (
                        0.5 * (3 - c_v) * u - 0.5 * (1 - c_v) * u_p1
                )

                switch_rh = 0.5 * (1. - tau / hx * v_avg) * u -\
                0.5 * (1. + tau / hx * v_avg) * u_n1

                return v_avg * (switch_lh * msk + switch_rh * (1 - msk))

        @staticmethod
        def DV3(u, v):

                u_p1 = np.pad(u, ((0, 0), (0, 1)), mode='constant')[:, 1:]

                v_p2 = np.pad(u, ((0, 2), (0, 0)), mode='constant')[2:]
                v_p1 = np.pad(u, ((0, 1), (0, 0)), mode='constant')[1:]
                v_n1 = np.pad(u, ((1, 0), (0, 0)), mode='constant')[:-1]

                u_avg = (u_p1 + u) / 2 
                c_u   = np.abs(u_avg) * tau / hx

                sign_value = u_avg * (v_p1 - v) * 0.5 * (v_p2 - v_p1 - v + v_n1)
                msk = sign_value >= 0 

                switch_lh = (u_avg >= 0) * (
                        0.5 * (3 - c_u) * v - 0.5 * (1 - c_u) * v_n1
                ) + (u_avg < 0) * (
                        0.5 * (3 - c_u) * v_p1 - 0.5 * (1 - c_u) * v_p2
                )

                switch_rh = 0.5 * (1. - tau / hx * u_avg) * v_p1 -\
                                        0.5 * (1. + tau / hx * u_avg) * v

                return u_avg * (switch_lh * msk + switch_rh * (1 - msk))

        @staticmethod
        def DV1(u, v):

                u    = np.pad(u, ((1, 0), (0, 0)), mode='constant')[1:] 
                u_p1 = np.pad(u, ((0, 0), (0, 1)), mode='constant')[:, 1:]

                v_p1 = np.pad(u, ((0, 0), (0, 1)), mode='constant')[:, 1:]
                v_n1 = np.pad(u, ((0, 0), (1, 0)), mode='constant')[:, :-1]
                v_n2 = np.pad(u, ((0, 0), (2, 0)), mode='constant')[:, :-2]

                u_avg = (u_p1 + u) / 2 
                c_u   = np.abs(u_avg) * tau / hx

                sign_value = u_avg * (v - v_n1) * 0.5 * (v_p1 - v - v_n1 + v_n2)
                msk = sign_value >= 0 
    
                switch_lh = (u_avg >= 0) * (
                        0.5 * (3 - c_u) * v_n1 - 0.5 * (1 - c_u) * v_n2
                ) + (u_avg < 0) * (
                        0.5 * (3 - c_u) * v - 0.5 * (1 - c_u) * v_p1
                )

                switch_rh = 0.5 * (1. - tau / hx * u_avg) * v -\
                                        0.5 * (1. + tau / hx * u_avg) * v_n1

                return u_avg * (switch_lh * msk + switch_rh * (1 - msk))

        @staticmethod
        def DV4(v):
    
                v_p2 = np.pad(v, ((0, 0), (0, 2)), mode='constant')[:, 2:]
                v_p1 = np.pad(v, ((0, 0), (0, 1)), mode='constant')[:, 1:]
                v_n1 = np.pad(v, ((0, 0), (1, 0)), mode='constant')[:, :-1]

                v_avg = (v_p1 + v) / 2 
                c_v   = np.abs(v_avg) * tau / hx

                sign_value = v_avg * (v_p1 - v) * 0.5 * (v_p2 - v_p1 - v + v_n1)
                msk = sign_value >= 0 

                switch_lh = (v_avg >= 0) * (
                        0.5 * (3 - c_v) * v - 0.5 * (1 - c_v) * v_n1
                ) + (v_avg < 0) * (
                        0.5 * (3 - c_v) * v_p1 - 0.5 * (1 - c_v) * v_p2
                )

                switch_rh = 0.5 * (1. - tau / hx * v_avg) * v_p1 -\
                                        0.5 * (1. + tau / hx * v_avg) * v

                return v_avg * (switch_lh * msk + switch_rh * (1 - msk))

        @staticmethod
        def DV2(v):
            
                v_p1 = np.pad(v, ((0, 0), (0, 1)), mode='constant')[:, 1:]
                v_n1 = np.pad(v, ((0, 0), (1, 0)), mode='constant')[:, :-1]
                v_n2 = np.pad(v, ((0, 0), (2, 0)), mode='constant')[:, :-2]

                v_avg = (v + v_n1) / 2 
                c_v   = np.abs(v_avg) * tau / hx

                sign_value = v_avg * (v - v_n1) * 0.5 * (v_p1 - v - v_n1 + v_n2)

                msk = sign_value >= 0 

                switch_lh = (v_avg >= 0) * (
                        0.5 * (3 - c_v) * v_n1 - 0.5 * (1 - c_v) * v_n2
                ) + (v_avg < 0) * (
                        0.5 * (3 - c_v) * v - 0.5 * (1 - c_v) * v_p1
                )

                switch_rh = 0.5 * (1. - tau / hx * v_avg) * v -\
                                        0.5 * (1. + tau / hx * v_avg) * v_n1

                return v_avg * (switch_lh * msk + switch_rh * (1 - msk))

        @staticmethod
        def DS3(u, s):
            
                u_n1 = np.pad(u, ((0, 0), (1, 0)), mode='constant')[:, :-1]

                s_p2 = np.pad(u, ((0, 2), (0, 0)), mode='constant')[2:]
                s_p1 = np.pad(u, ((0, 1), (0, 0)), mode='constant')[1:]
                s_n1 = np.pad(u, ((1, 0), (0, 0)), mode='constant')[:-1]

                u_avg = (u + u_n1) / 2 
                c_u   = np.abs(u_avg) * tau / hx

                sign_value = u_avg * (s_p1 - s) * 0.5 * (s_p2 - s_p1 - s + s_n1)

                msk = sign_value >= 0 

                switch_lh = (u_avg >= 0) * (
                        0.5 * (3 - c_u) * s - 0.5 * (1 - c_u) * s_n1
                ) + (u_avg < 0) * (
                        0.5 * (3 - c_u) * s_p1 - 0.5 * (1 - c_u) * s_p2
                )

                switch_rh = 0.5 * (1. - tau / hx * u_avg) * s_p1 -\
                                        0.5 * (1. + tau / hx * u_avg) * s

                return u_avg * (switch_lh * msk + switch_rh * (1 - msk))

        @staticmethod
        def DS1(u, s):
    
                u    = np.pad(u, ((1, 0), (0, 0)), mode='constant')[:-1]
                u_n1 = np.pad(u, ((0, 0), (1, 0)), mode='constant')[:, :-1]

                s_p1 = np.pad(u, ((0, 1), (0, 0)), mode='constant')[1:]
                s_n1 = np.pad(u, ((1, 0), (0, 0)), mode='constant')[:-1]
                s_n2 = np.pad(u, ((2, 0), (0, 0)), mode='constant')[:-2]

                u_avg = (u + u_n1) / 2 
                c_u   = np.abs(u_avg) * tau / hx

                sign_value = u_avg * (s - s_n1) * 0.5 * (s_p1 - s - s_n1 + s_n2)

                msk = sign_value >= 0 

                switch_lh = (u_avg >= 0) * (
                        0.5 * (3 - c_u) * s_n1 - 0.5 * (1 - c_u) * s_n2
                ) + (u_avg < 0) * (
                        0.5 * (3 - c_u) * s - 0.5 * (1 - c_u) * s_p1
                )

                switch_rh = 0.5 * (1. - tau / hx * u_avg) * s -\
                                        0.5 * (1. + tau / hx * u_avg) * s_n1

                return u_avg * (switch_lh * msk + switch_rh * (1 - msk))
    
        @staticmethod    
        def DS4(v, s):
    
                v_p1 = np.pad(v, ((0, 0), (0, 1)), mode='constant')[:, 1:]

                s_p2 = np.pad(s, ((0, 0), (0, 2)), mode='constant')[:, 2:]
                s_p1 = np.pad(s, ((0, 0), (0, 1)), mode='constant')[:, 1:]
                s_n1 = np.pad(s, ((0, 0), (1, 0)), mode='constant')[:, :-1]

                v_avg = (v_p1 + v) / 2 
                c_v   = np.abs(v_avg) * tau / hx

                sign_value = v_avg * (s_p1 - s) * 0.5 * (s_p2 - s_p1 - s + s_n1)
                msk = sign_value >= 0 

                switch_lh = (v_avg >= 0) * (
                        0.5 * (3 - c_v) * s - 0.5 * (1 - c_v) * s_n1
                ) + (v_avg < 0) * (
                        0.5 * (3 - c_v) * s_p1 - 0.5 * (1 - c_v) * s_p2
                )

                switch_rh = 0.5 * (1. - tau / hx * v_avg) * s_p1 -\
                                        0.5 * (1. + tau / hx * v_avg) * s

                return v_avg * (switch_lh * msk + switch_rh * (1 - msk))

        @staticmethod
        def DS2(v_old, s_old):
    
                v_n1 = np.pad(v, ((0, 0), (1, 0)), mode='constant')[:, :-1]

                s_p1 = np.pad(s, ((0, 0), (0, 1)), mode='constant')[:, 1:]
                s_n1 = np.pad(s, ((0, 0), (1, 0)), mode='constant')[:, :-1]
                s_n2 = np.pad(s, ((0, 0), (2, 0)), mode='constant')[:, :-2]

                v_avg = (v + v_n1) / 2 
                c_v   = np.abs(v_avg) * tau / hx

                sign_value = v_avg * (s - s_n1) * 0.5 * (s_p1 - s - s_n1 + s_n2)

                msk = sign_value >= 0 

                switch_lh = (v_avg >= 0) * (
                        0.5 * (3 - c_v) * s_n1 - 0.5 * (1 - c_v) * s_n2
                ) + (v_avg < 0) * (
                        0.5 * (3 - c_v) * s - 0.5 * (1 - c_v) * s_p1
                )

                switch_rh = 0.5 * (1. - tau / hx * v_avg) * s -\
                                        0.5 * (1. + tau / hx * v_avg) * s_n1

                return v_avg * (switch_lh * msk + switch_rh * (1 - msk))


        def u_tilde(self, u_old, v_old):

                hx  = self.hx
                hy  = self.hy
                tau = self.tau
                Re  = self.Re
                Fr  = self.Fr

                u = np.copy(u_old)
                v = np.copy(v_old)

                v_p1_x = np.pad(v, ((0, 1), (0, 0)), mode='constant')[1:]
                v_n1_y = np.pad(v, ((0, 0), (1, 0)), mode='constant')[:, :-1]
                v_n1_y_p1_x = np.pad(v_n1_y, ((0, 1), (0, 0)), mode='constant')[1:]

                u_p1_y = np.pad(u, ((0, 0), (0, 1)), mode='constant')[:, 1:]
                u_n1_y = np.pad(u, ((0, 0), (1, 0)), mode='constant')[:, :-1]

                u_ = u_old
                u_ += tau / hx * (self.DU3(u) - self.DU1(u))
                u_ += tau / hy * (self.DU4(u, v) - self.DU2(u, v))

                u_ -= tau / hy / Re * (
                        ((v_p1_x - v) / hx - (u_p1_y - u) / hy) -\
                        ((v_n1_y_p1_x - v_n1_y) / hx - (u - u_n1_y) / hy)
                )

                return u_

        def v_tilde(self, u_old, v_old, s_old):

                hx  = self.hx
                hy  = self.hy
                tau = self.tau
                Re  = self.Re
                Fr  = self.Fr

                u = np.copy(u_old)
                v = np.copy(v_old)

                v_p1_x = np.pad(v, ((0, 1), (0, 0)), mode='constant')[1:]
                v_n1_x = np.pad(v, ((1, 0), (0, 0)), mode='constant')[:-1]

                u_p1_y = np.pad(u, ((0, 0), (0, 1)), mode='constant')[:, 1:]
                u_n1_x = np.pad(u, ((1, 0), (0, 0)), mode='constant')[:-1]
                u_n1_x_p1_y = np.pad(u_n1_x, ((0, 0), (0, 1)), mode='constant')[:, 1:]

                 
                v_ = np.copy(v_old)
                v_ += tau / hy * (self.DV4(u) - self.DV2(u))
                v_ += tau / hx * (self.DV3(u, v) - self.DV1(u, v))
                v_ += tau / hx / Re * (
                        ((v_p1_x - v) / hx - (u_p1_y - u) / hy) -\
                        ((v - v_n1_x) / hx - (u_n1_x_p1_y - u_n1_x) / hy)
                )

                v_ += -tau / Fr * s_old

                return v_

        def s_tilde(self, s_old, u_old, v_old):

                hx  = self.hx
                hy  = self.hy
                tau = self.tau
                Re  = self.Re
                Fr  = self.Fr
                Sc  = self.Sc
                C   = self.C

                s = np.copy(s_old)
                u = np.copy(u_old)
                v = np.copy(v_old)

                s_p1_x = np.pad(s, ((0, 1), (0, 0)), mode='constant')[1:]
                s_n1_x = np.pad(s, ((1, 0), (0, 0)), mode='constant')[:-1]
                s_p1_y = np.pad(s, ((0, 0), (0, 1)), mode='constant')[:, 1:]
                s_n1_y = np.pad(s, ((0, 0), (1, 0)), mode='constant')[:, :-1]

                s_ = s - tau / hx * (self.DS3(u_next, s) - self.DS1(u_next, s)) \
                           - tau / hy * (self.DS4(v_next, s) - self.DS2(v_next, s)) \
               + tau / (Sc * Re) * ((s_p1_x - 2.* s + s_n1_x) / hx**2 + (s_p1_y - 2.* s + s_n1_y) / hy**2) \
               + tau * v_next / C


        def div_tilde(self, u_tilde, v_tilde):

                hx  = self.hx
                hy  = self.hy
                tau = self.tau

                u_p1_x = np.pad(u_tilde, ((0, 1), (0, 0)), mode='constant')[1:]
                u_n1_x = np.pad(u_tilde, ((1, 0), (0, 0)), mode='constant')[:-1]

                v_p1_y = np.pad(v_tilde, ((0, 0), (0, 1)), mode='constant')[:, 1:]
                v_n1_y = np.pad(v_tilde, ((0, 0), (1, 0)), mode='constant')[:, :-1]

                return (u_p1_x - u_n1_x) / (2 * hx) + (v_p1_y - v_n1_y) / (2 * hy) 

        def step(self, u_old, v_old, s_old):

                hx  = self.hx
                hy  = self.hy
                tau = self.tau

                u_ = self.u_tilde(u_old, v_old)
                v_ = self.v_tilde(u_old, v_old)

                p  = self.solve_poisson()

                u_next = u_ - tau / hx * (np.pad(p, ((0, 1), (0, 0)), mode='constant')[1:] - p)
                v_next = v_ - tau / hy * (np.pad(p, ((0, 0), (0, 1)), mode='constant')[:, 1:] - p)

                s_next = self.s_tilde(s_old, u_next, v_next)

                return u_next, v_next, s_next


        def solve_poisson(self):
            p = PoissonProblem(p_init=self.p, rhs=self.div_tilde(u_, v_) / self.tau, hx=self.hx, hy=self.hy)
            p.solve(**poisson_solver_defaults)

            return p.get_solution
