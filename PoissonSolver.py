import numpy as np



class PoissonProblem:
    
    @staticmethod
    def generate_chebyshev_index_sequence(r: int = 5):
        prev_idx = [1, 2]
        for _r in range(2, r + 1):
            cur_idx = [_v for _i in prev_idx for _v in (_i, 2**_r + 1 - _i)]
            prev_idx = cur_idx
        
        cur_idx = np.array(cur_idx, dtype=int)

        return cur_idx

    def __init__(self, p_init, rhs, h_x: float, h_y: float):
        assert(rhs.shape == p_init.shape)

        self.h_x = h_x
        self.h_y = h_y

        self.n_x = p_init.shape[0] + 2
        self.n_y = p_init.shape[1] + 2

        self.solution = p_init.copy()
        self.chebyshev_steps = None

        self.rhs = rhs
        self.err = np.inf
        self.err_history = []

    def residual(self):
        s = np.pad(self.solution, [(1, 1), (1, 1)], mode='edge') # So that BC dp/dn = 0 holds
        res = (s[0:-2, 1:-1] - 2.0*s[1:-1, 1:-1] + s[2:, 1:-1])/self.h_x**2 \
                                       + (s[1:-1, 0:-2] - 2.0*s[1:-1, 1:-1] + s[1:-1, 2:])/self.h_y**2 \
                                       - self.rhs
        self.err = np.linalg.norm(res.flatten(), ord=np.inf)
        self.err_history.append(self.err)
        return res

    def step(self, tau: np.float64):
        self.solution += tau*self.residual()
    
    def set_chebyshev_steps(self, batch_size: int = 5):
        # l and L are operator's minimal and maximal eigvals
        l = 4*(np.sin(np.pi/2/self.n_x)**2/self.h_x**2 + np.sin(np.pi/2/self.n_y)**2/self.h_y**2)
        L = 4*(np.cos(np.pi/2/self.n_x)**2/self.h_x**2 + np.cos(np.pi/2/self.n_y)**2/self.h_y**2)
        mu = L/l
        n_iter = 2**batch_size
        _is = PoissonProblem.generate_chebyshev_index_sequence(batch_size)
        taus = 1./((L + l)/2 + (L - l)/2*np.cos(np.pi*(2*_is - 1)/2/n_iter))
        self.chebyshev_steps = taus

    def iterate_chebyshev(self, eps):
        for i, tau in enumerate(self.chebyshev_steps):
            self.step(tau)
            if self.err <= eps:
                break
        return i

    def solve(self, eps: np.float64, batch_size: int = 5, max_iter: int = 100000):
        self.set_chebyshev_steps(batch_size)
        total_iter = 0
        while self.err > eps:
            total_iter += self.iterate_chebyshev(eps)
            if total_iter >= max_iter:
                break
                #raise(RuntimeWarning(f'Residual is {self.err:.2e} after {total_iter:d} steps, probably instability'))
        return total_iter

    def get_solution(self,):
        return self.solution
