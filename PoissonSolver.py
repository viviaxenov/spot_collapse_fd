import numpy as np



class PoissonProblem:

    def fill_ghost_cells(self,):
        # set p[-1, i] = p[0, i] and so on so the BC dp/dn = 0 holds
        self.solution[0, :] = self.solution[1, :]
        self.solution[-1, :] = self.solution[-2, :]
        self.solution[:, 0] = self.solution[:, 1]
        self.solution[:, -1] = self.solution[:, -2]

        return self.solution

    def __init__(self, p_init, rhs, h_x: float, h_y: float):

        assert(rhs.shape == p_init.shape)

        self.h_x = h_x
        self.h_y = h_y

        self.n_x = p_init.shape[0] + 2
        self.n_y = p_init.shape[1] + 2

        self.solution = p_init.copy()

        self.rhs = rhs
        self.err = np.inf

    def residual(self, w_error: bool = False):
        s = np.pad(self.solution, [(1, 1), (1, 1)], mode='edge')
        res = (s[0:-2, 1:-1] - 2.0*s[1:-1, 1:-1] + s[2:, 1:-1])/self.h_x**2 \
                                       + (s[1:-1, 0:-2] - 2.0*s[1:-1, 1:-1] + s[1:-1, 2:])/self.h_y**2 \
                                       - self.rhs
        if w_error:
            self.err = np.linalg.norm(res.flatten(), ord=np.inf)
        return res

    def step(self, tau: np.float64, w_error: bool = False):
        self.solution += tau*self.residual(w_error)

    def iterate_chebyshev(self, n_iter: int = None, eps: np.float64=1e-3):
        # operator's minimal and maximal eigvals
        l = 4*(np.sin(np.pi/2/self.n_x)**2/self.h_x**2 + np.sin(np.pi/2/self.n_y)**2/self.h_y**2)
        L = 4*(np.cos(np.pi/2/self.n_x)**2/self.h_x**2 + np.cos(np.pi/2/self.n_y)**2/self.h_y**2)
        mu = L/l

        if n_iter is None:
            n_iter = int(np.ceil(-np.sqrt(mu)/2*np.log(eps))) + 1

        for i in range(1, n_iter + 1):
            tau = 1./((L + l)/2 + (L - l)/2*np.cos(np.pi*(2*i - 1)/2/n_iter))
            self.step(tau, w_error=(i == n_iter))
        return n_iter

    def solve(self, n_iter: int, eps: np.float64, max_iter=100000):
        total_iter = 0
        while self.err > eps:
            total_iter += self.iterate_chebyshev(n_iter=n_iter)
            if total_iter >= max_iter:
                break
                #raise(RuntimeWarning(f'Residual is {self.err:.2e} after {total_iter:d} steps, probably instability'))
        return total_iter

    def get_solution(self,):
        return self.solution[1:-1, 1:-1]
