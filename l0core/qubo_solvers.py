"""
QUBO solvers and utilities for L0 optimization.

This module provides QUBO-based solvers for L0-regularized sparse signal recovery,
including interfaces to external QUBO solvers and hybrid approaches.
"""

import numpy as np
import jax
import jax.numpy as jnp

try:
    from eqc_models.solvers import Dirac3CloudSolver, Dirac3IntegerCloudSolver, Dirac3ContinuousCloudSolver
    from eqc_models.base import QuadraticModel
    EQC_MODELS_AVAILABLE = True
except ImportError:
    EQC_MODELS_AVAILABLE = False
    # Define dummy classes if not available
    class QuadraticModel:
        def __init__(self, C, J):
            self.C = C
            self.J = J
            self.upper_bound = None
            print("Dummy QuadraticModel initialized. Real QCI solvers are not available.")
    class Dirac3IntegerCloudSolver:
        def solve(self, model, num_samples=1, sum_constraint=None, relaxation_schedule=1):
            print("Dummy Dirac3IntegerCloudSolver: Solve called, returning empty solution.")
            return {"results": {"solutions": [np.zeros(len(model.C) if hasattr(model,'C') else 0).tolist()] 
                               if (hasattr(model,'C') and len(model.C)>0) else [] }}

from l0core.hamiltonians import hamiltonian_h_prime

def solve_alternating_relaxed_mip_external_qubo(
        A_cs_matrix_np,       # Sensing matrix (M_measure, N_signal)
        y_cs_vec_np,          # Measurement vector (M_measure,)
        lambda_param,         # Lambda for H' and for T_i in ICM
        N_signal,             # Number of signal components (N_r_dim)
        M_measure,            # Number of measurements (M_k_dim)
        num_outer_iter=20,
        num_R_iter_coord=10,   # Iterations for R update using coordinate descent
        qubo_num_samples=10,
        qubo_relaxation_schedule=1,
        initial_R_np=None,
        initial_x_np=None,    # Initial x (continuous 0 to 1)
        verbose=True,
        random_seed=42):
    """
    Solves the MIP problem by alternating between:
    1. Optimizing R (continuous) given x (binary), using coordinate descent on least squares.
    2. Optimizing x (binary) given R (continuous), using an external QUBO solver on H'.
    
    Parameters:
    -----------
    A_cs_matrix_np : numpy.ndarray
        Sensing matrix of shape (M_measure, N_signal)
    y_cs_vec_np : numpy.ndarray
        Measurement vector of shape (M_measure,)
    lambda_param : float
        Regularization parameter lambda
    N_signal : int
        Dimension of signal
    M_measure : int
        Dimension of measurements
    num_outer_iter : int, optional
        Number of outer iterations
    num_R_iter_coord : int, optional
        Number of iterations for updating R using coordinate descent
    qubo_num_samples : int, optional
        Number of samples for the QUBO solver
    qubo_relaxation_schedule : int, optional
        Relaxation schedule parameter for the QUBO solver
    initial_R_np : numpy.ndarray, optional
        Initial value for R
    initial_x_np : numpy.ndarray, optional
        Initial value for x
    verbose : bool, optional
        Whether to print progress information
    random_seed : int, optional
        Random seed for initialization
        
    Returns:
    --------
    tuple : (R_final, x_final, history_H_prime)
    """
    if not EQC_MODELS_AVAILABLE:
        print("Error: Dirac3CloudSolver/QuadraticModel not available. Cannot run with External QUBO.")
        print("Returning initialized values.")
        return (initial_R_np if initial_R_np is not None else np.zeros(N_signal)), \
               (initial_x_np if initial_x_np is not None else np.zeros(N_signal)), []

    np.random.seed(random_seed)
    
    # Initialize R_current and x_current (continuous 0 to 1)
    if initial_R_np is not None:
        R_current_np = np.copy(initial_R_np)
    else:
        R_current_np = np.random.randn(N_signal) * 0.1

    if initial_x_np is not None:
        x_current_np = np.clip(np.copy(initial_x_np), 0.0, 1.0)
    else:
        x_current_np = (np.abs(R_current_np) > 1e-3).astype(float)
    
    # For coordinate descent (R update)
    A_cs_T_np = A_cs_matrix_np.T
    P_matrix_np_for_coord = A_cs_T_np @ A_cs_matrix_np
    At_y_np_for_coord = A_cs_T_np @ y_cs_vec_np
    
    # For QUBO (x update) based on H'
    P_matrix_np_for_qubo = P_matrix_np_for_coord  # Same matrix
    At_y_np_for_qubo = At_y_np_for_coord  # Same vector
    
    # For calculating H' after each update (monitoring)
    A_input_matrix_jax_for_H_calc = jnp.array(A_cs_matrix_np.T)
    y_data_jax_for_H_calc = jnp.array(y_cs_vec_np)

    # Initialize solver
    dirac_solver = Dirac3IntegerCloudSolver()

    history_H_prime = []

    if verbose:
        print(f"\n--- Starting Alternating Relaxed MIP Solver (External QUBO, R_coord) ---")
        print(f"Lambda={lambda_param}, OuterIter={num_outer_iter}, R_coord_iter={num_R_iter_coord}")
        print(f"QUBO Samples={qubo_num_samples}, QUBO RelaxSched={qubo_relaxation_schedule}")

    for outer_it in range(num_outer_iter):
        # --- 1. Update R given x (using Coordinate Descent on Least Squares) ---
        for r_it in range(num_R_iter_coord):
            for i in range(N_signal):
                if x_current_np[i] < 0.5:  # If x_i is effectively 0, R_i should be 0
                    R_current_np[i] = 0.0
                    continue
                
                # We're minimizing ||y - A diag(x) R||^2 with respect to R_i
                # This is equivalent to minimizing ||y - sum_j A_j x_j R_j||^2
                # where A_j is the j-th column of A.
                # For each R_i, the optimum is:
                # R_i = (A_i^T (y - sum_{j!=i} A_j x_j R_j)) / (A_i^T A_i)
                # Since x_i are binary, we can simply use active columns
                
                # Current prediction without R_i's contribution
                curr_s = R_current_np * x_current_np
                pred_without_i = A_cs_matrix_np @ curr_s - A_cs_matrix_np[:, i] * curr_s[i]
                residual = y_cs_vec_np - pred_without_i
                
                numerator = np.dot(A_cs_matrix_np[:, i], residual)
                denominator = np.dot(A_cs_matrix_np[:, i], A_cs_matrix_np[:, i])
                
                if denominator > 1e-9:
                    R_current_np[i] = numerator / denominator
                else:
                    R_current_np[i] = 0.0
        
        # --- 2. Update x given R (using external QUBO solver on H') ---
        # QUBO formulation for H': 0.5 * s^T P s - q^T s + lambda * sum(x_i)
        # where s = diag(R) x, P = A^T A, q = A^T y
        # This gives a quadratic form in x: C^T x + x^T J x where
        # C_i = 0.5 * R_i^2 * P_ii - R_i * q_i + lambda
        # J_ij = R_i * R_j * P_ij / 2.0 (for i≠j)
        
        C_model = np.zeros(N_signal)
        J_model_for_qubo = np.zeros((N_signal, N_signal))

        for i in range(N_signal):
            C_model[i] = (0.5 * P_matrix_np_for_qubo[i,i] * (R_current_np[i]**2)) + \
                            lambda_param - \
                            (R_current_np[i] * At_y_np_for_qubo[i])
            for j in range(i + 1, N_signal):
                val_P_ij_Ri_Rj = P_matrix_np_for_qubo[i,j] * R_current_np[i] * R_current_np[j]
                J_model_for_qubo[i,j] = val_P_ij_Ri_Rj / 2.0
                J_model_for_qubo[j,i] = val_P_ij_Ri_Rj / 2.0
        
        qubo_problem = QuadraticModel(C_model.astype(np.float64), J_model_for_qubo.astype(np.float64))
        qubo_problem.upper_bound = np.ones((N_signal,), dtype=np.int64)

        try:
            response = dirac_solver.solve(qubo_problem, 
                                         num_samples=qubo_num_samples, 
                                         # sum_constraint=N_signal*0.3,  # Optional constraint 
                                         relaxation_schedule=qubo_relaxation_schedule)
            if response and "results" in response and "solutions" in response["results"] and response["results"]["solutions"]:
                x_new_from_qubo = np.array(response["results"]["solutions"][0], dtype=float)
                if x_new_from_qubo.shape == x_current_np.shape:
                    x_current_np = x_new_from_qubo
                else:
                    if verbose: print(f"Warning: QUBO solver returned x of unexpected shape. Keeping previous x.")
            else:
                if verbose: print(f"Warning: QUBO solver did not return valid solutions. Keeping previous x.")
        except Exception as e:
            if verbose: print(f"Error during QUBO solve: {e}. Keeping previous x.")
        
        # Ensure consistency: R_i = 0 if x_i = 0
        R_current_np[x_current_np < 0.5] = 0.0
        
        # Calculate H' for monitoring
        current_H_prime_val = hamiltonian_h_prime(jnp.array(R_current_np), jnp.array(x_current_np),
                                                 A_input_matrix_jax_for_H_calc, y_data_jax_for_H_calc, lambda_param)
        history_H_prime.append(float(current_H_prime_val))
        
        if verbose and (outer_it % (num_outer_iter // 10 if num_outer_iter > 10 else 1) == 0 or outer_it == num_outer_iter - 1):
            sparsity_x = np.sum(x_current_np > 0.5)
            recon_error = 0.5 * np.mean((y_cs_vec_np - A_cs_matrix_np @ (R_current_np * x_current_np))**2)
            print(f"Iter {outer_it:3d}: Sparsity_x = {sparsity_x}, H' = {current_H_prime_val:.4f}, ReconError = {recon_error:.4e}")

    if verbose: print("Alternating Relaxed MIP Solver (External QUBO) finished.")
    return R_current_np, x_current_np, history_H_prime

def solve_alternating_relaxed_mip_external_qubo_R_CG(
        A_cs_matrix_np,        # Sensing matrix (M_measure, N_signal)
        y_cs_vec_np,           # Measurement vector (M_measure,)
        lambda_param,          # Lambda for H' and for T_i in ICM
        N_signal,              # Number of signal components
        M_measure,             # Number of measurements
        num_outer_iter=20,
        num_R_iter_cg=10,      # Iterations for R update using Conjugate Gradient
        cg_tolerance=1e-7,     # Tolerance for CG convergence
        qubo_num_samples=10,
        qubo_relaxation_schedule=1,
        initial_R_np=None,
        initial_x_np=None,     # Initial x (continuous 0 to 1)
        verbose=True,
        random_seed=42):
    """
    Solves the MIP problem by alternating between:
    1. Optimizing R (continuous) given x (binary), using Conjugate Gradient on least squares.
    2. Optimizing x (binary) given R (continuous), using an external QUBO solver on H'.
    
    Parameters:
    -----------
    A_cs_matrix_np : numpy.ndarray
        Sensing matrix of shape (M_measure, N_signal)
    y_cs_vec_np : numpy.ndarray
        Measurement vector of shape (M_measure,)
    lambda_param : float
        Regularization parameter lambda
    N_signal : int
        Dimension of signal
    M_measure : int
        Dimension of measurements
    num_outer_iter : int, optional
        Number of outer iterations
    num_R_iter_cg : int, optional
        Number of iterations for updating R using Conjugate Gradient
    cg_tolerance : float, optional
        Tolerance for CG convergence
    qubo_num_samples : int, optional
        Number of samples for the QUBO solver
    qubo_relaxation_schedule : int, optional
        Relaxation schedule parameter for the QUBO solver
    initial_R_np : numpy.ndarray, optional
        Initial value for R
    initial_x_np : numpy.ndarray, optional
        Initial value for x
    verbose : bool, optional
        Whether to print progress information
    random_seed : int, optional
        Random seed for initialization
        
    Returns:
    --------
    tuple : (R_final, x_final, history_H_prime)
    """
    if not EQC_MODELS_AVAILABLE:
        print("Error: Dirac3CloudSolver/QuadraticModel not available. Cannot run with External QUBO.")
        print("Returning initialized values.")
        return (initial_R_np if initial_R_np is not None else np.zeros(N_signal)), \
               (initial_x_np if initial_x_np is not None else np.zeros(N_signal)), []

    np.random.seed(random_seed)
    
    # Initialize R_current and x_current
    if initial_R_np is not None:
        R_current_np = np.copy(initial_R_np)
    else:
        R_current_np = np.random.randn(N_signal) * 0.1

    if initial_x_np is not None:
        x_current_np = np.clip(np.copy(initial_x_np), 0.0, 1.0).astype(float)
    else:
        x_current_np = (np.abs(R_current_np) > 1e-3).astype(float)
    
    # For QUBO (x update) based on H'
    P_matrix_np_for_qubo = A_cs_matrix_np.T @ A_cs_matrix_np
    At_y_np_for_qubo = A_cs_matrix_np.T @ y_cs_vec_np
    
    # For calculating H' after each update (monitoring)
    A_input_matrix_jax_for_H_calc = jnp.array(A_cs_matrix_np.T)
    y_data_jax_for_H_calc = jnp.array(y_cs_vec_np)

    # Initialize solver
    dirac_solver = Dirac3IntegerCloudSolver()

    history_H_prime = []

    if verbose:
        print(f"\n--- Starting Alternating Relaxed MIP Solver (External QUBO, R_CG) ---")
        print(f"Lambda={lambda_param}, OuterIter={num_outer_iter}, R_CG_iter={num_R_iter_cg}, CG_tol={cg_tolerance}")
        print(f"QUBO Samples={qubo_num_samples}, QUBO RelaxSched={qubo_relaxation_schedule}")

    for outer_it in range(num_outer_iter):
        # --- 1. Update R given x (using Conjugate Gradient on Least Squares) ---
        # We only need to solve for the active components where x_i = 1
        active_indices = np.where(x_current_np > 0.5)[0]
        
        if len(active_indices) > 0:
            A_active = A_cs_matrix_np[:, active_indices] # Shape (M_measure, N_active)
            R_active = R_current_np[active_indices]      # Shape (N_active,)
            
            # Solve (A_active^T A_active) R_active = A_active^T y_cs_vec_np for R_active
            
            # Initial gradient g = A_active^T (A_active R_active - y_cs_vec_np)
            grad_g = A_active.T @ (A_active @ R_active - y_cs_vec_np)
            
            p_cg = -grad_g # Initial search direction
            r_cg_sq_old = np.dot(grad_g, grad_g)

            for cg_iter in range(num_R_iter_cg):
                if np.sqrt(r_cg_sq_old) < cg_tolerance: # Check convergence based on gradient norm
                    break
                
                # Compute A_active^T A_active @ p_cg efficiently using matrix-vector products
                Ap_cg = A_active @ p_cg # (M_measure,)
                Q_eff_p_cg = A_active.T @ Ap_cg # (N_active,)
                
                alpha_cg_denominator = np.dot(p_cg, Q_eff_p_cg)
                if abs(alpha_cg_denominator) < 1e-12: # Avoid division by zero 
                    break
                
                alpha_cg = r_cg_sq_old / alpha_cg_denominator
                
                R_active = R_active + alpha_cg * p_cg
                grad_g = grad_g + alpha_cg * Q_eff_p_cg # Update gradient
                
                r_cg_sq_new = np.dot(grad_g, grad_g)
                beta_cg = r_cg_sq_new / r_cg_sq_old
                p_cg = -grad_g + beta_cg * p_cg
                r_cg_sq_old = r_cg_sq_new
            
            R_current_np[active_indices] = R_active
        # R_i for inactive x_i (x_i=0) should be 0
        R_current_np[x_current_np < 0.5] = 0.0
        
        # --- 2. Update x given R (using external QUBO solver on H') ---
        C_model = np.zeros(N_signal)
        J_model_for_qubo = np.zeros((N_signal, N_signal))

        for i in range(N_signal):
            C_model[i] = (0.5 * P_matrix_np_for_qubo[i,i] * (R_current_np[i]**2)) + \
                           lambda_param - \
                           (R_current_np[i] * At_y_np_for_qubo[i])
            for j in range(i + 1, N_signal):
                val_P_ij_Ri_Rj = P_matrix_np_for_qubo[i,j] * R_current_np[i] * R_current_np[j]
                J_model_for_qubo[i,j] = val_P_ij_Ri_Rj / 2.0 
                J_model_for_qubo[j,i] = val_P_ij_Ri_Rj / 2.0 
        
        qubo_problem = QuadraticModel(C_model.astype(np.float64), J_model_for_qubo.astype(np.float64))
        qubo_problem.upper_bound = np.ones((N_signal,), dtype=np.int64)

        try:
            response = dirac_solver.solve(qubo_problem, 
                                          num_samples=qubo_num_samples, 
                                          # sum_constraint = N_signal*0.6, # Optional constraint
                                          relaxation_schedule=qubo_relaxation_schedule)
            if response and "results" in response and "solutions" in response["results"] and response["results"]["solutions"]:
                x_new_from_qubo = np.array(response["results"]["solutions"][0], dtype=float)
                if x_new_from_qubo.shape == x_current_np.shape:
                     x_current_np = x_new_from_qubo
                else:
                    if verbose: print(f"Warning: QUBO solver returned x of unexpected shape. Keeping previous x.")
            else:
                if verbose: print("Warning: QUBO solver did not return valid solutions. Keeping previous x.")
        except Exception as e:
            if verbose: print(f"Error during QUBO solve: {e}. Keeping previous x.")
        
        # Ensure consistency: R_i = 0 if x_i = 0
        R_current_np[x_current_np < 0.5] = 0.0

        # Calculate H' for monitoring
        current_H_prime_val = hamiltonian_h_prime(jnp.array(R_current_np), jnp.array(x_current_np),
                                                  A_input_matrix_jax_for_H_calc, y_data_jax_for_H_calc, lambda_param)
        history_H_prime.append(float(current_H_prime_val))

        if verbose and (num_outer_iter < 10 or outer_it % (num_outer_iter // 10) == 0 or outer_it == num_outer_iter -1) :
            sparsity_x = np.sum(x_current_np > 0.5)
            recon_error = 0.5 * np.mean((y_cs_vec_np - A_cs_matrix_np @ (R_current_np * x_current_np))**2)
            print(f"RelaxedMIP (Ext QUBO, R_CG) Iter {outer_it:3d}: Sparsity_x = {sparsity_x}, "
                  f"H' = {current_H_prime_val:.4f}, ReconError = {recon_error:.4e}")

    if verbose: print("Alternating Relaxed MIP Solver (External QUBO, R_CG) finished.")
    return R_current_np, x_current_np, history_H_prime

def solve_l0_hybrid_R_CG_x_QUBO(
        A_matrix,       # Sensing matrix (M, N)
        y_vector,       # Measurement vector (M,)
        lambda_val,     # Sparsity regularization parameter
        N_signal,       # Number of signal components
        num_outer_iter=30,
        num_R_iter_cg=10,     # Iterations for R update using Conjugate Gradient
        cg_tolerance=1e-7,    # Tolerance for CG convergence
        qubo_num_samples=10,
        qubo_relaxation_schedule=1,
        initial_R=None,
        initial_x=None,
        verbose=True):
    """
    Solves L0-regularized problem: min 0.5 ||A s - y||^2 + lambda ||s||_0
    by alternating between:
    1. Optimizing R (values of s) given x (support of s) using Conjugate Gradient.
    2. Optimizing x (support of s) given R (values of s) using an external QUBO solver.
    Here, R effectively represents the signal s.
    
    Parameters:
    -----------
    A_matrix : numpy.ndarray
        Sensing matrix of shape (M, N)
    y_vector : numpy.ndarray
        Measurement vector of shape (M,)
    lambda_val : float
        Regularization parameter lambda
    N_signal : int
        Dimension of signal
    num_outer_iter : int, optional
        Number of outer iterations
    num_R_iter_cg : int, optional
        Number of iterations for updating R using Conjugate Gradient
    cg_tolerance : float, optional
        Tolerance for CG convergence
    qubo_num_samples : int, optional
        Number of samples for the QUBO solver
    qubo_relaxation_schedule : int, optional
        Relaxation schedule parameter for the QUBO solver
    initial_R : numpy.ndarray, optional
        Initial value for R
    initial_x : numpy.ndarray, optional
        Initial value for x
    verbose : bool, optional
        Whether to print progress information
        
    Returns:
    --------
    tuple : (R_final, x_final)
    """
    if not EQC_MODELS_AVAILABLE:
        print("Error: Dirac3CloudSolver/QuadraticModel not available. Cannot run L0 Hybrid solver.")
        return np.zeros(N_signal), np.zeros(N_signal)

    # Initialization
    if initial_R is not None:
        R_current = np.copy(initial_R)
    else:
        R_current = np.linalg.lstsq(A_matrix, y_vector, rcond=None)[0] # Initial R from least squares

    if initial_x is not None:
        x_current = (np.copy(initial_x) > 0.5).astype(float) # Ensure binary
    else:
        x_current = (np.abs(R_current) > 1e-3).astype(float) # Initial x based on R's magnitude
    
    R_current[x_current < 0.5] = 0.0 # Ensure consistency: R_i = 0 if x_i = 0

    # Precompute for QUBO (A^T A and A^T y)
    AtA = A_matrix.T @ A_matrix
    Aty = A_matrix.T @ y_vector
    
    dirac_solver = Dirac3IntegerCloudSolver() 

    if verbose:
        print(f"\n--- Starting L0 Hybrid Solver (R via Conj.Grad., x via External QUBO) ---")
        print(f"Lambda={lambda_val}, OuterIter={num_outer_iter}, R_CG_iter={num_R_iter_cg}, CG_tol={cg_tolerance}")
        print(f"QUBO Samples={qubo_num_samples}, QUBO RelaxSched={qubo_relaxation_schedule}")

    for outer_it in range(num_outer_iter):
        # --- 1. Update R (signal values) given x (support) using Conjugate Gradient ---
        active_indices = np.where(x_current > 0.5)[0]
        
        R_active_prev_iter = R_current[active_indices] # Store for comparison if needed

        if len(active_indices) > 0:
            A_active = A_matrix[:, active_indices]    # Shape (M, N_active)
            R_active = R_current[active_indices]      # Initial guess for R_active (shape N_active)
            
            # Initial gradient g = A_active^T (A_active R_active - y_vector)
            grad_g = A_active.T @ (A_active @ R_active - y_vector)
            
            p_cg = -grad_g # Initial search direction
            r_cg_sq_old = np.dot(grad_g, grad_g)

            if np.sqrt(r_cg_sq_old) < cg_tolerance and outer_it > 0 : # Already converged for R
                 if verbose and outer_it % 5 == 0 : 
                     print(f"  Iter {outer_it:2d} (R-CG): Converged early for R_active. Grad norm: {np.sqrt(r_cg_sq_old):.2e}")
                 pass # Skip CG iterations if gradient is already small
            else:
                for cg_iter in range(num_R_iter_cg):
                    if np.sqrt(r_cg_sq_old) < cg_tolerance:
                        if verbose and outer_it % 5 == 0: 
                            print(f"  Iter {outer_it:2d} (R-CG): R_active converged in {cg_iter} CG iterations. Grad norm: {np.sqrt(r_cg_sq_old):.2e}")
                        break
                    
                    # Matrix-vector product for CG: Q_eff_p_cg = (A_active^T A_active) @ p_cg
                    Ap_cg = A_active @ p_cg 
                    Q_eff_p_cg = A_active.T @ Ap_cg
                    
                    alpha_cg_denominator = np.dot(p_cg, Q_eff_p_cg)
                    if abs(alpha_cg_denominator) < 1e-12: 
                        if verbose and outer_it % 5 == 0: 
                            print(f"  Iter {outer_it:2d} (R-CG): alpha_cg denominator too small at CG iter {cg_iter}.")
                        break 
                    
                    alpha_cg = r_cg_sq_old / alpha_cg_denominator
                    
                    R_active = R_active + alpha_cg * p_cg
                    grad_g = grad_g + alpha_cg * Q_eff_p_cg 
                    
                    r_cg_sq_new = np.dot(grad_g, grad_g)
                    beta_cg = r_cg_sq_new / r_cg_sq_old
                    p_cg = -grad_g + beta_cg * p_cg
                    r_cg_sq_old = r_cg_sq_new
                
                R_current[active_indices] = R_active
        # R_i for inactive x_i (x_i=0) should be 0
        R_current[x_current < 0.5] = 0.0
        
        # --- 2. Update x (support) given R (signal values) using external QUBO solver ---
        # Objective for x: 0.5 ||A diag(R) x - y||^2 + lambda * sum(x_i)
        # QUBO formulation: sum_i C_i x_i + sum_{i<j} J_ij x_i x_j
        # C_i = 0.5 * R_i^2 * (A^T A)_{ii} - R_i * (A^T y)_i + lambda
        # J_ij (for sum_{i<j}) = R_i * R_j * (A^T A)_{ij}
        # For QuadraticModel(C_qm, J_qm_symm) where E = C_qm^T x + x^T J_qm_symm x:
        # C_qm_i = C_i
        # J_qm_symm_ij = J_ij / 2.0 for i!=j
        
        C_model_qubo = np.zeros(N_signal)
        J_model_qubo_symmetric = np.zeros((N_signal, N_signal))

        for i in range(N_signal):
            C_model_qubo[i] = (0.5 * AtA[i,i] * (R_current[i]**2)) + \
                              lambda_val - \
                              (R_current[i] * Aty[i])
            for j in range(i + 1, N_signal):
                val_AtA_ij_Ri_Rj = AtA[i,j] * R_current[i] * R_current[j]
                J_model_qubo_symmetric[i,j] = val_AtA_ij_Ri_Rj / 2.0 
                J_model_qubo_symmetric[j,i] = val_AtA_ij_Ri_Rj / 2.0 
        
        qubo_problem = QuadraticModel(C_model_qubo.astype(np.float64), J_model_qubo_symmetric.astype(np.float64))
        qubo_problem.upper_bound = np.ones((N_signal,), dtype=np.int64)

        x_prev_iter = np.copy(x_current) # Store for comparison

        try:
            response = dirac_solver.solve(qubo_problem, 
                                          num_samples=qubo_num_samples, 
                                          # sum_constraint = N_signal*0.3, # Optional constraint
                                          relaxation_schedule=qubo_relaxation_schedule)
            if response and "results" in response and "solutions" in response["results"] and response["results"]["solutions"]:
                x_new_from_qubo = np.array(response["results"]["solutions"][0], dtype=float)
                if x_new_from_qubo.shape == x_current.shape:
                     x_current = x_new_from_qubo
                else:
                    if verbose: print(f"  Iter {outer_it:2d} (x-QUBO): QUBO solver returned x of unexpected shape. Keeping previous x.")
            else:
                if verbose: print(f"  Iter {outer_it:2d} (x-QUBO): QUBO solver did not return valid solutions. Keeping previous x.")
        except Exception as e:
            if verbose: print(f"  Iter {outer_it:2d} (x-QUBO): Error during QUBO solve: {e}. Keeping previous x.")
        
        R_current[x_current < 0.5] = 0.0 # Ensure R_i = 0 if x_i became 0

        # Check for convergence (e.g., if x and R didn't change much)
        current_sparsity = np.sum(x_current > 0.5)
        if verbose and (outer_it % 5 == 0 or outer_it == num_outer_iter -1) :
            mse_val = np.mean((A_matrix @ (R_current * x_current) - y_vector)**2)
            l0_norm = np.sum(x_current)
            obj_val = 0.5 * mse_val * A_matrix.shape[0] + lambda_val * l0_norm # Rescale mse to sum of squares
            print(f"Iter {outer_it:2d}: Sparsity(x)={current_sparsity}, Recon_MSE={mse_val:.4e}, Obj={obj_val:.4e}")
        
        if outer_it > 0 and np.all(x_current == x_prev_iter):
            if verbose: print(f"  Iter {outer_it:2d}: x support converged. Stopping.")
            break

    if verbose: print("L0 Hybrid Solver finished.")
    return R_current, x_current
