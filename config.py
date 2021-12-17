import numpy as np

# field of integration
Y = 5.0
X = 15.0

# spot parameters
spot_center_x = 0.0
spot_center_y = 0.0
R_0 = 1.0

# target time
T = 1.0

# other constants
N = 1.0
Tb = 2 * np.pi / N

mu_div_rho = 0.01
k_s = 1.41e-5
C = 10.0

Re = 100.0  # Reynolds number
Fr = 0.1  # Froude number
Sc = 709.2  # Schmidt number

poisson_solver_defaults = {"max_iter": 5000, "batch_size": 7, "eps": 1e-4}
