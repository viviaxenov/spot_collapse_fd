import numpy as np

# field of integration
Y = 0.
X = 15.

# spot parameters
spot_center_x = 0.
spot_center_y = 0.
R_0           = 1.

# target time
T = 1.

# other constants

g      = 9.81 # acceleration of gravity 
rho_0  = 1.
Lambda = 10.
N      = 1.
Tb     = 2 * np.pi  / N

mu_div_rho = 0.01
k_s        = 1.41e-5
C          = 10.

Re = 100.   # Reynolds number 
Fr = 0.1    # Froude number 
Sc = 709.2  # Schmidt number

poisson_solver_defaults = {'max_iter': 10000, 'batch_size': 7, 'eps': 1e-3}
