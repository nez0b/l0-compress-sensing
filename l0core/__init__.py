# L0 Optimization Library for Compressed Sensing
# This package contains implementations of various L0-optimization methods
# for sparse signal recovery in the compressed sensing framework.

__version__ = "0.1.0"

# Import key functions for convenient access
from .transforms import (
    haar2_transform_flat,
    ihaar2_transform_flat,
    inverse_haar2_transform_from_flat,
    dct2_transform_flat,
    idct2_transform_flat,
    inverse_dct2_transform_flat
)

from .hamiltonians import (
    hamiltonian_h_prime_single_var,
    calculate_h_prime_relaxed,
    calculate_user_hamiltonian
)

from .penalties import (
    penalty_terms_single_var,
    total_loss_function_single_var
)

from .solvers import (
    solve_continuous_relaxation_single_var,
    solve_continuous_relaxation_sgd,
    solve_alternating_relaxed_h_prime,
    solve_alternating_mip
)

# Optional QUBO solvers if dependencies are available
try:
    from .qubo_solvers import (
        prep_qubo_relaxed_hamiltonian,
        solve_qubo_relaxed,
        hybrid_solver
    )
except ImportError:
    pass
