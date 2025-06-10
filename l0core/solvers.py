"""
Solvers for L0 optimization in compressed sensing.

This module provides various solver implementations for L0-regularized
sparse signal recovery, including JAX-based continuous relaxations and
alternating minimization approaches.
"""

import numpy as np
import jax
import jax.numpy as jnp

from l0core.hamiltonians import (
    hamiltonian_h_prime, 
    hamiltonian_h_prime_single_var,
    calculate_h_prime_relaxed,
    calculate_user_hamiltonian
)
from l0core.penalties import (
    penalty_terms_single_var,
    penalty_terms_original,
    total_loss_function_single_var,
    total_loss_function_original_hamiltonian
)

# --- JAX-based continuous relaxation solvers ---

# Gradient function using JAX for single variable approach
grad_loss_fn_single_var = jax.jit(jax.grad(total_loss_function_single_var, argnums=(0, 1)))

# Optimization step for single variable approach
@jax.jit
def update_step_single_var(R_vars, x_vars, grads, learning_rate):
    """
    Performs one gradient descent update step and clips x_vars.
    
    Parameters:
    -----------
    R_vars : jnp.ndarray
        Vector R of shape (N_r_dim,)
    x_vars : jnp.ndarray
        Vector x of shape (N_r_dim,)
    grads : tuple of jnp.ndarray
        Gradients for R_vars and x_vars
    learning_rate : float
        Learning rate for gradient descent update
        
    Returns:
    --------
    tuple : Updated (R_new, x_clipped)
    """
    grad_R, grad_x = grads

    R_new = R_vars - learning_rate * grad_R
    x_new = x_vars - learning_rate * grad_x

    # Clip x to [0,1]
    x_clipped = jnp.clip(x_new, 0.0, 1.0)

    return R_new, x_clipped

def solve_continuous_relaxation_single_var(
        A_sensing_matrix_np_transposed, 
        y_measurement_vec_np,       
        lambda_param,
        N_r_dim,                      
        M_k_dim,                      
        learning_rate=0.01,
        num_iterations=1000,
        penalty_coeff=1.0,
        random_seed=42,
        verbose=True):
    """
    Solves the continuous relaxation using single variable approach.
    
    Parameters:
    -----------
    A_sensing_matrix_np_transposed : numpy.ndarray
        Transpose of the sensing matrix, shape (N_r_dim, M_k_dim)
    y_measurement_vec_np : numpy.ndarray
        Measurement vector, shape (M_k_dim,)
    lambda_param : float
        Regularization parameter lambda
    N_r_dim : int
        Dimension of R and x variables
    M_k_dim : int
        Dimension of measurement vector
    learning_rate : float, optional
        Learning rate for gradient descent
    num_iterations : int, optional
        Number of iterations to run
    penalty_coeff : float, optional
        Weight coefficient for the binary constraint penalty
    random_seed : int, optional
        Random seed for initialization
    verbose : bool, optional
        Whether to print progress information
        
    Returns:
    --------
    tuple : (R_final, x_final, loss_history)
    """
    key = jax.random.PRNGKey(random_seed)
    key_R, key_x = jax.random.split(key)

    # R_vars can be positive or negative, initialized around zero.
    R_vars = jax.random.normal(key_R, (N_r_dim,)) * 0.1 
    # x_vars initialized in [0,1]
    x_vars = jax.random.uniform(key_x, (N_r_dim,))

    A_jax = jnp.array(A_sensing_matrix_np_transposed)
    y_data_jax = jnp.array(y_measurement_vec_np)

    loss_history = []

    if verbose:
        print(f"\n--- Starting JAX L0 Optimization (Single Variable Approach) ---")
        print(f"N_r_dim (signal)={N_r_dim}, M_k_dim (measurements)={M_k_dim}, Lambda={lambda_param}")
        print(f"LR={learning_rate}, Iterations={num_iterations}")
        print(f"Penalty (x(1-x)=0)={penalty_coeff}")

    for iteration in range(num_iterations):
        grads = grad_loss_fn_single_var(R_vars, x_vars, A_jax, y_data_jax, 
                                      lambda_param, penalty_coeff)
        
        R_vars, x_vars = update_step_single_var(R_vars, x_vars, grads, learning_rate)

        if verbose and (iteration % (num_iterations // 10) == 0 or iteration == num_iterations - 1):
            current_loss = total_loss_function_single_var(
                R_vars, x_vars, A_jax, y_data_jax, lambda_param, penalty_coeff
            )
            loss_history.append(float(current_loss))
            binary_error = jnp.sum((x_vars * (1.0 - x_vars))**2)
            sparsity = jnp.sum(x_vars > 0.5)
            s_current = R_vars * x_vars
            recon_error_term = 0.5 * jnp.sum((y_data_jax - A_jax.T @ s_current)**2)
            print(f"Iter {iteration:5d}: Loss = {current_loss:.4f}, ReconErrTerm = {recon_error_term:.4f}, "
                  f"Binary_err = {binary_error:.4e}, Sparsity (x>0.5) = {sparsity}")
                  
    if verbose: print("JAX L0 Optimization (Single Variable) finished.")
    return R_vars, x_vars, loss_history

# Stochastic Gradient Descent implementation

def compute_stochastic_gradient(R_vars, x_vars, A_jax, y_data_jax, lambda_param, penalty_coeff, batch_size, key):
    """
    Compute gradient for a random subset of variables.
    
    Parameters:
    -----------
    R_vars : jnp.ndarray
        Vector R of shape (N_r_dim,)
    x_vars : jnp.ndarray
        Vector x of shape (N_r_dim,)
    A_jax : jnp.ndarray
        Matrix A, shape (N_r_dim, M_k_dim)
    y_data_jax : jnp.ndarray
        Vector y, shape (M_k_dim,)
    lambda_param : float
        Regularization parameter lambda
    penalty_coeff : float
        Weight coefficient for the binary constraint penalty
    batch_size : int
        Size of the random subset
    key : jax.random.PRNGKey
        Random key for selection
        
    Returns:
    --------
    tuple : (stochastic_grad_R, stochastic_grad_x)
    """
    # Select random indices for the batch
    N_r_dim = R_vars.shape[0]
    indices = jax.random.choice(key, N_r_dim, shape=(batch_size,), replace=False)
    
    # Create a mask to select the batch elements
    mask_R = jnp.zeros_like(R_vars)
    mask_X = jnp.zeros_like(x_vars)
    mask_R = mask_R.at[indices].set(1.0)
    mask_X = mask_X.at[indices].set(1.0)
    
    # Get the full gradients
    grad_R, grad_x = jax.grad(total_loss_function_single_var, argnums=(0, 1))(
        R_vars, x_vars, A_jax, y_data_jax, lambda_param, penalty_coeff
    )
    
    # Return only the gradients for the selected batch elements
    stochastic_grad_R = grad_R * mask_R
    stochastic_grad_x = grad_x * mask_X
    
    return stochastic_grad_R, stochastic_grad_x

def update_step_sgd(R_vars, x_vars, grads, learning_rate, momentum=0.9, velocity_R=None, velocity_x=None):
    """
    Performs one SGD update step with momentum and clips x_vars.
    
    Parameters:
    -----------
    R_vars : jnp.ndarray
        Vector R of shape (N_r_dim,)
    x_vars : jnp.ndarray
        Vector x of shape (N_r_dim,)
    grads : tuple of jnp.ndarray
        Gradients for R_vars and x_vars
    learning_rate : float
        Learning rate for update
    momentum : float, optional
        Momentum coefficient
    velocity_R : jnp.ndarray, optional
        Velocity vector for R
    velocity_x : jnp.ndarray, optional
        Velocity vector for x
        
    Returns:
    --------
    tuple : (R_new, x_clipped, velocity_R, velocity_x)
    """
    grad_R, grad_x = grads
    
    # Initialize velocity if it doesn't exist
    if velocity_R is None:
        velocity_R = jnp.zeros_like(R_vars)
    if velocity_x is None:
        velocity_x = jnp.zeros_like(x_vars)
    
    # Update velocity with momentum
    velocity_R = momentum * velocity_R - learning_rate * grad_R
    velocity_x = momentum * velocity_x - learning_rate * grad_x
    
    # Update parameters
    R_new = R_vars + velocity_R
    x_new = x_vars + velocity_x

    # Clip x to [0,1]
    x_clipped = jnp.clip(x_new, 0.0, 1.0)

    return R_new, x_clipped, velocity_R, velocity_x

def solve_continuous_relaxation_sgd(
        A_sensing_matrix_np_transposed, 
        y_measurement_vec_np,       
        lambda_param,
        N_r_dim,                      
        M_k_dim,                      
        learning_rate=0.01,
        num_iterations=1000,
        penalty_coeff=1.0,
        batch_size=None,  # If None, use 10% of dimensions
        momentum=0.9,     # Momentum coefficient
        random_seed=42,
        verbose=True):
    """
    Solves the continuous relaxation using SGD with momentum.
    
    Parameters:
    -----------
    A_sensing_matrix_np_transposed : numpy.ndarray
        Transpose of the sensing matrix, shape (N_r_dim, M_k_dim)
    y_measurement_vec_np : numpy.ndarray
        Measurement vector, shape (M_k_dim,)
    lambda_param : float
        Regularization parameter lambda
    N_r_dim : int
        Dimension of R and x variables
    M_k_dim : int
        Dimension of measurement vector
    learning_rate : float, optional
        Learning rate for SGD
    num_iterations : int, optional
        Number of iterations to run
    penalty_coeff : float, optional
        Weight coefficient for the binary constraint penalty
    batch_size : int, optional
        Batch size for SGD, defaults to 10% of dimensions if None
    momentum : float, optional
        Momentum coefficient for SGD
    random_seed : int, optional
        Random seed for initialization
    verbose : bool, optional
        Whether to print progress information
        
    Returns:
    --------
    tuple : (R_final, x_final, loss_history)
    """
    key = jax.random.PRNGKey(random_seed)
    key_R, key_x, key_init = jax.random.split(key, 3)

    # R_vars can be positive or negative, initialized around zero.
    R_vars = jax.random.normal(key_R, (N_r_dim,)) * 0.1 
    # x_vars initialized in [0,1]
    x_vars = jax.random.uniform(key_x, (N_r_dim,))

    # Initialize velocities for momentum
    velocity_R = jnp.zeros_like(R_vars)
    velocity_x = jnp.zeros_like(x_vars)

    A_jax = jnp.array(A_sensing_matrix_np_transposed)
    y_data_jax = jnp.array(y_measurement_vec_np)

    loss_history = []
    
    # Set batch size if not specified
    if batch_size is None:
        batch_size = max(int(N_r_dim * 0.1), 1)  # Default to 10% of dimensions

    if verbose:
        print(f"\n--- Starting JAX L0 Optimization (SGD with Momentum) ---")
        print(f"N_r_dim (signal)={N_r_dim}, M_k_dim (measurements)={M_k_dim}, Lambda={lambda_param}")
        print(f"LR={learning_rate}, Iterations={num_iterations}, Batch Size={batch_size}, Momentum={momentum}")
        print(f"Penalty (x(1-x)=0)={penalty_coeff}")

    for iteration in range(num_iterations):
        # Get a new random key for this iteration
        key_iter = jax.random.fold_in(key_init, iteration)
        
        # Compute stochastic gradients
        stochastic_grads = compute_stochastic_gradient(
            R_vars, x_vars, A_jax, y_data_jax, lambda_param, penalty_coeff, batch_size, key_iter
        )
        
        # Update parameters with SGD + momentum
        R_vars, x_vars, velocity_R, velocity_x = update_step_sgd(
            R_vars, x_vars, stochastic_grads, learning_rate, 
            momentum=momentum, velocity_R=velocity_R, velocity_x=velocity_x
        )

        if verbose and (iteration % (num_iterations // 10) == 0 or iteration == num_iterations - 1):
            current_loss = total_loss_function_single_var(
                R_vars, x_vars, A_jax, y_data_jax, lambda_param, penalty_coeff
            )
            loss_history.append(float(current_loss))
            binary_error = jnp.sum((x_vars * (1.0 - x_vars))**2)
            sparsity = jnp.sum(x_vars > 0.5)
            s_current = R_vars * x_vars
            recon_error_term = 0.5 * jnp.sum((y_data_jax - A_jax.T @ s_current)**2)
            print(f"Iter {iteration:5d}: Loss = {current_loss:.4f}, ReconErrTerm = {recon_error_term:.4f}, "
                  f"Binary_err = {binary_error:.4e}, Sparsity (x>0.5) = {sparsity}")
                  
    if verbose: print("JAX L0 Optimization (SGD with Momentum) finished.")
    return R_vars, x_vars, loss_history

# Gradient function using JAX for original Hamiltonian
grad_loss_fn_original_h = jax.jit(jax.grad(total_loss_function_original_hamiltonian, argnums=(0, 1, 2)))

# Optimization step for original Hamiltonian
@jax.jit
def update_step_original_h(R_vars, x_vars, y_internal_vars, grads, learning_rate):
    """
    Performs one gradient descent update step and clips x_vars, y_internal_vars.
    
    Parameters:
    -----------
    R_vars : jnp.ndarray
        Vector R of shape (N_r_dim,)
    x_vars : jnp.ndarray
        Vector x of shape (N_r_dim,)
    y_internal_vars : jnp.ndarray
        Vector y of shape (N_r_dim,)
    grads : tuple of jnp.ndarray
        Gradients for R_vars, x_vars, and y_internal_vars
    learning_rate : float
        Learning rate for gradient descent update
        
    Returns:
    --------
    tuple : (R_new, x_clipped, y_internal_clipped)
    """
    grad_R, grad_x, grad_y_internal = grads

    R_new = R_vars - learning_rate * grad_R
    x_new = x_vars - learning_rate * grad_x
    y_internal_new = y_internal_vars - learning_rate * grad_y_internal

    x_clipped = jnp.clip(x_new, 0.0, 1.0) 
    y_internal_clipped = jnp.clip(y_internal_new, 0.0, 1.0)

    return R_new, x_clipped, y_internal_clipped

def solve_continuous_relaxation_original_hamiltonian(
        A_sensing_matrix_np_transposed,
        y_measurement_vec_np,
        lambda_param,
        N_r_dim,
        M_k_dim,
        learning_rate=0.01,
        num_iterations=1000,
        penalty_coeff1=1.0,
        penalty_coeff2=1.0,
        random_seed=42,
        verbose=True):
    """
    Solves the continuous relaxation using the user's original Hamiltonian H'.
    
    Parameters:
    -----------
    A_sensing_matrix_np_transposed : numpy.ndarray
        Transpose of the sensing matrix, shape (N_r_dim, M_k_dim)
    y_measurement_vec_np : numpy.ndarray
        Measurement vector, shape (M_k_dim,)
    lambda_param : float
        Regularization parameter lambda
    N_r_dim : int
        Dimension of R and x variables
    M_k_dim : int
        Dimension of measurement vector
    learning_rate : float, optional
        Learning rate for gradient descent
    num_iterations : int, optional
        Number of iterations to run
    penalty_coeff1 : float, optional
        Weight coefficient for the x_r + y_r = 1 constraint
    penalty_coeff2 : float, optional
        Weight coefficient for the x_r * y_r = 0 constraint
    random_seed : int, optional
        Random seed for initialization
    verbose : bool, optional
        Whether to print progress information
        
    Returns:
    --------
    tuple : (R_final, x_final, y_internal_final, loss_history)
    """
    key = jax.random.PRNGKey(random_seed)
    key_R, key_x, key_y = jax.random.split(key, 3)

    # R_vars can be positive or negative, initialized around zero.
    R_vars = jax.random.normal(key_R, (N_r_dim,)) * 0.1 
    # x_vars initialized in [0,1]
    x_vars = jax.random.uniform(key_x, (N_r_dim,))
    # y_internal_vars initialized such that x+y is close to 1 initially
    y_internal_vars = 1.0 - x_vars 

    A_jax = jnp.array(A_sensing_matrix_np_transposed)
    y_data_jax = jnp.array(y_measurement_vec_np)

    loss_history = []

    if verbose:
        print(f"\n--- Starting JAX L0 Optimization (Modified Stable Hamiltonian) ---")
        print(f"N_r_dim (signal)={N_r_dim}, M_k_dim (measurements)={M_k_dim}, Lambda={lambda_param}")
        print(f"LR={learning_rate}, Iterations={num_iterations}")
        print(f"Penalty1 (x+y=1)={penalty_coeff1}, Penalty2 (xy=0)={penalty_coeff2}")

    for iteration in range(num_iterations):
        grads = grad_loss_fn_original_h(R_vars, x_vars, y_internal_vars,
                                        A_jax, y_data_jax, lambda_param,
                                        penalty_coeff1, penalty_coeff2)
        
        R_vars, x_vars, y_internal_vars = update_step_original_h(
            R_vars, x_vars, y_internal_vars, grads, learning_rate
        )

        # Print progress if requested
        if verbose and (num_iterations // 10 == 0 or iteration % (num_iterations // 10) == 0 or iteration == num_iterations - 1):
            current_loss = total_loss_function_original_hamiltonian(
                R_vars, x_vars, y_internal_vars, A_jax, y_data_jax, 
                lambda_param, penalty_coeff1, penalty_coeff2
            )
            loss_history.append(float(current_loss))
            c1_error = jnp.sum((x_vars + y_internal_vars - 1.0)**2)
            c2_error = jnp.sum((x_vars * y_internal_vars)**2)
            sparsity = jnp.sum(x_vars > 0.5)
            s_current = R_vars * x_vars
            recon_error_term = 0.5 * jnp.sum((y_data_jax - A_jax.T @ s_current)**2)
            print(f"Iter {iteration:5d}: Loss = {current_loss:.4f}, ReconErrTerm = {recon_error_term:.4f}, "
                  f"C1_err = {c1_error:.4e}, C2_err = {c2_error:.4e}, Sparsity (x>0.5) = {sparsity}")
                  
    if verbose: print("JAX L0 Optimization (Modified Hamiltonian) finished.")
    return R_vars, x_vars, y_internal_vars, loss_history

# --- Alternating minimization solvers ---

def solve_alternating_relaxed_h_prime(
        A_cs_matrix, y_cs_vec, lambda_h_prime,
        N_signal, M_measure,
        num_outer_iter=50, num_s_update_iter=5, num_x_update_iter=5,
        penalty_coeff_x=1.0, learning_rate_x=0.01,
        initial_R=None, initial_x=None, verbose=True, convergence_tol=1e-5):
    """
    Solves the relaxed MIP problem by alternating between optimizing R and x.
    - R is updated based on minimizing ||y_cs - A_cs @ (R*x)||_2^2 by first solving for s = R*x.
    - x is updated by gradient descent on H' + penalty_for_x(1-x)=0.
    H' = sum_{r<r'} P_rr' (R_r x_r) (R_r' x_r') - sum_r q_r (R_r x_r) + lambda sum_r x_r
    
    Parameters:
    -----------
    A_cs_matrix : numpy.ndarray
        Sensing matrix of shape (M_measure, N_signal)
    y_cs_vec : numpy.ndarray
        Measurement vector of shape (M_measure,)
    lambda_h_prime : float
        Regularization parameter lambda
    N_signal : int
        Dimension of signal
    M_measure : int
        Dimension of measurements
    num_outer_iter : int, optional
        Number of outer iterations
    num_s_update_iter : int, optional
        Number of iterations for updating s (and R)
    num_x_update_iter : int, optional
        Number of iterations for updating x
    penalty_coeff_x : float, optional
        Weight coefficient for the x(1-x)=0 penalty
    learning_rate_x : float, optional
        Learning rate for x gradient updates
    initial_R : numpy.ndarray, optional
        Initial value for R
    initial_x : numpy.ndarray, optional
        Initial value for x
    verbose : bool, optional
        Whether to print progress information
    convergence_tol : float, optional
        Tolerance for convergence check
    
    Returns:
    --------
    tuple : (R_final, x_final, history_loss)
    """
    if initial_R is not None:
        R_current = np.copy(initial_R)
    else:
        R_current = np.random.randn(N_signal) * 0.1
    
    if initial_x is not None:
        x_current = np.copy(initial_x).astype(float)
    else:
        # Initialize x to be somewhat sparse but not all zero/one
        x_current = np.random.uniform(0.2, 0.8, N_signal) 
        # x_current = np.ones(N_signal) * 0.5

    # Precompute P = A_cs.T @ A_cs and q = A_cs.T @ y_cs
    P_matrix = A_cs_matrix.T @ A_cs_matrix
    q_vector = A_cs_matrix.T @ y_cs_vec

    if verbose:
        print(f"\n--- Starting Alternating Relaxed H' Solver ---")
        print(f"Lambda_H'={lambda_h_prime}, OuterIter={num_outer_iter}")
        print(f"S_update_iter={num_s_update_iter}, X_update_iter={num_x_update_iter}")
        print(f"Penalty_x(1-x)={penalty_coeff_x}, LR_x={learning_rate_x}")

    history_loss = []
    s_current = R_current * x_current

    for outer_it in range(num_outer_iter):
        R_old = np.copy(R_current)
        x_old = np.copy(x_current)

        # 1. Update s = R*x (then R) based on least squares: min ||y_cs - A_cs @ s||_2^2
        #    where s_i = 0 if x_i is effectively 0.
        for s_it in range(num_s_update_iter):
            for i in range(N_signal):
                if np.abs(x_current[i]) < 1e-6: # If x_i is zero, s_i must be zero
                    s_current[i] = 0.0
                    continue

                # Calculate current prediction without s_i's contribution
                # A_cs @ s_current - A_cs_matrix[:, i] * s_current[i]
                s_others_contribution = np.dot(A_cs_matrix, s_current) - A_cs_matrix[:, i] * s_current[i]
                residual_for_si = y_cs_vec - s_others_contribution
                
                numerator = np.dot(A_cs_matrix[:, i], residual_for_si)
                denominator = np.dot(A_cs_matrix[:, i], A_cs_matrix[:, i])

                if denominator > 1e-9:
                    s_current[i] = numerator / denominator
                else:
                    s_current[i] = 0.0
        
        # Update R from s and x
        for i in range(N_signal):
            if np.abs(x_current[i]) < 1e-6:
                R_current[i] = 0.0 # Or keep previous R_current[i] if x_current[i] is tiny but not zero
            else:
                R_current[i] = s_current[i] / x_current[i]
        
        # 2. Update x given R (Gradient Descent for H' + penalty)
        for x_it in range(num_x_update_iter):
            for s_idx in range(N_signal):
                # Gradient of H' w.r.t x_s_idx
                # dH'/dx_s = R_s * ( sum_{j!=s} P_sj R_j x_j - q_s ) + lambda
                sum_P_R_x_j_neq_s = 0.0
                for j_idx in range(N_signal):
                    if s_idx == j_idx:
                        continue
                    sum_P_R_x_j_neq_s += P_matrix[s_idx, j_idx] * R_current[j_idx] * x_current[j_idx]
                
                grad_H_prime_xs = R_current[s_idx] * (sum_P_R_x_j_neq_s - q_vector[s_idx]) + lambda_h_prime
                
                # Gradient of penalty_coeff_x * (x_s(1-x_s))^2 w.r.t x_s
                # penalty = p * (x - x^2)^2 => d/dx = p * 2 * (x - x^2) * (1 - 2x)
                xs = x_current[s_idx]
                penalty_gradient_xs = penalty_coeff_x * 2.0 * (xs - xs**2) * (1.0 - 2.0*xs)
                
                total_gradient_xs = grad_H_prime_xs + penalty_gradient_xs
                
                x_current[s_idx] = x_current[s_idx] - learning_rate_x * total_gradient_xs
                x_current[s_idx] = np.clip(x_current[s_idx], 0.0, 1.0)

        # Update s_current for the next iteration or for calculating H'
        s_current = R_current * x_current

        # Optional: Calculate and store loss for monitoring
        current_h_prime = calculate_h_prime_relaxed(A_cs_matrix, y_cs_vec, R_current, x_current, lambda_h_prime)
        current_penalty_val = penalty_coeff_x * np.sum((x_current * (1.0 - x_current))**2)
        current_total_loss = current_h_prime + current_penalty_val
        history_loss.append(current_total_loss)

        if verbose and (outer_it % (num_outer_iter // 10 if num_outer_iter >=10 else 1) == 0 or outer_it == num_outer_iter -1) :
            sparsity_x = np.sum(x_current > 0.5)
            # For reconstruction error, use ||y_cs - A_cs s||^2
            recon_error = 0.5 * np.sum((y_cs_vec - A_cs_matrix @ s_current)**2)
            print(f"Iter {outer_it:3d}: Loss = {current_total_loss:.4f} (H'={current_h_prime:.4f}, Pen={current_penalty_val:.4e}), "
                  f"ReconErr_s = {recon_error:.4f}, Sparsity(x>0.5) = {sparsity_x}")
        
        # Convergence check
        delta_R = np.linalg.norm(R_current - R_old) / (np.linalg.norm(R_old) + 1e-9)
        delta_x = np.linalg.norm(x_current - x_old) / (np.linalg.norm(x_old) + 1e-9)
        if delta_R < convergence_tol and delta_x < convergence_tol and outer_it > 10:
            if verbose: print(f"Converged at iteration {outer_it}.")
            break

    if verbose: print("Alternating Relaxed H' Solver finished.")
    return R_current, x_current, history_loss

def solve_alternating_mip(A_cs_matrix, y_cs_vec, lambda_mip, N_signal, M_measure,
                          num_outer_iter=50, num_R_iter=5, num_sigma_iter=5,
                          initial_R=None, initial_sigma=None, verbose=True):
    """
    Solves the MIP problem by alternating between optimizing R and sigma.
    R is updated using coordinate descent for the least squares problem.
    sigma is updated using ICM for the QUBO defined by the user's Hamiltonian.
    
    Parameters:
    -----------
    A_cs_matrix : numpy.ndarray
        Sensing matrix of shape (M_measure, N_signal)
    y_cs_vec : numpy.ndarray
        Measurement vector of shape (M_measure,)
    lambda_mip : float
        Regularization parameter lambda
    N_signal : int
        Dimension of signal
    M_measure : int
        Dimension of measurements
    num_outer_iter : int, optional
        Number of outer iterations
    num_R_iter : int, optional
        Number of iterations for updating R
    num_sigma_iter : int, optional
        Number of iterations for updating sigma
    initial_R : numpy.ndarray, optional
        Initial value for R
    initial_sigma : numpy.ndarray, optional
        Initial value for sigma
    verbose : bool, optional
        Whether to print progress information
    
    Returns:
    --------
    tuple : (R_final, sigma_final, history_H_orig)
    """
    if initial_R is not None:
        R_current = np.copy(initial_R)
    else:
        R_current = np.random.randn(N_signal) * 0.1
     
    if initial_sigma is not None:
        sigma_current = np.copy(initial_sigma).astype(float)
    else:
        sigma_current = np.ones(N_signal) # Start with all sigma = 1

    # A_cs_matrix is (M_measure, N_signal), the sensing matrix
    # A_cs_T is (N_signal, M_measure), this is A_H in derivation.
    A_cs_T = A_cs_matrix.T 
    # P_matrix = A_H @ A_H.T = A_cs.T @ A_cs, shape (N_signal, N_signal)
    # P_ij = sum_k A_i^k A_j^k
    P_matrix = A_cs_T @ A_cs_matrix 
    # A_cs_T_y_cs = A_H @ y_H = A_cs.T @ y_cs, shape (N_signal,)
    # i-th element is sum_k A_i^k y^k
    A_cs_T_y_cs = A_cs_T @ y_cs_vec 

    if verbose:
        print(f"\n--- Starting Alternating MIP Solver (Original Binary Sigma) ---")
        print(f"Lambda_MIP={lambda_mip}, OuterIter={num_outer_iter}, R_iter={num_R_iter}, Sigma_iter={num_sigma_iter}")

    history_H_orig = [] # To store original Hamiltonian values

    for outer_it in range(num_outer_iter):
        # 1. Update R given sigma (Coordinate Descent for least squares)
        # Objective: min_R ||y_cs - A_cs @ diag(sigma) @ R||^2
        # This is equivalent to min_R ||y_cs - (A_cs * sigma_current_broadcasted) @ R||^2
        # Or, for each R_i where sigma_i=1: R_i = ( A_cs[:,i]^T * (y_cs - sum_{j!=i} A_cs[:,j] sigma_j R_j) ) / ||A_cs[:,i]||^2
        for r_it in range(num_R_iter):
            for i in range(N_signal):
                if sigma_current[i] > 0.5: # If sigma_i is effectively 1
                    # Calculate current prediction without R_i's contribution
                    current_s_vector = R_current * sigma_current
                    prediction_error_without_i = y_cs_vec - (A_cs_matrix @ current_s_vector - A_cs_matrix[:, i] * R_current[i] * sigma_current[i])
                    
                    numerator = np.dot(A_cs_matrix[:, i], prediction_error_without_i)
                    denominator = np.dot(A_cs_matrix[:, i], A_cs_matrix[:, i])

                    if denominator > 1e-9: 
                        R_current[i] = numerator / denominator
                    else:
                        R_current[i] = 0.0 
                else: # sigma_i is 0
                    R_current[i] = 0.0
         
        # 2. Update sigma given R (ICM for QUBO from user's Hamiltonian H)
        # H = sum_{r<r'} sum_k A_r^k A_r'^k R_r R_r' sigma_r sigma_r' - sum_r sum_k y^k A_r^k R_r sigma_r + lambda sum_r sigma_r
        # Cost to change sigma_i: Delta_H_i = sigma_i_new * T_i - sigma_i_old * T_i
        # We flip sigma_i if T_i < 0 (for sigma_i=1) or T_i > 0 (for sigma_i=0)
        # T_i = R_i * ( sum_{j!=i} (sum_k A_i^k A_j^k) R_j sigma_j - (sum_k y^k A_i^k) ) + lambda
        # T_i = R_i * ( sum_{j!=i} P_matrix[i,j] R_j sigma_j - A_cs_T_y_cs[i] ) + lambda_mip
        for s_it in range(num_sigma_iter):
            changed_in_pass = False
            for i in range(N_signal):
                if abs(R_current[i]) < 1e-9: # If R_i is (close to) zero, its sigma choice only driven by lambda
                    T_i = lambda_mip 
                else:
                    sum_P_R_sigma_j_neq_i = 0
                    for j in range(N_signal):
                        if i == j: continue
                        sum_P_R_sigma_j_neq_i += P_matrix[i, j] * R_current[j] * sigma_current[j]
                     
                    linear_term_coeff_i = A_cs_T_y_cs[i]
                    T_i = R_current[i] * (sum_P_R_sigma_j_neq_i - linear_term_coeff_i) + lambda_mip
                 
                new_sigma_i = 1.0 if T_i < 0 else 0.0 # if T_i < 0, setting sigma_i=1 is better
                if abs(sigma_current[i] - new_sigma_i) > 0.5: # If sigma_i flips
                    sigma_current[i] = new_sigma_i
                    changed_in_pass = True
            if not changed_in_pass and s_it > 0: 
                break 
         
        if verbose and (outer_it % (num_outer_iter // 10 if num_outer_iter >=10 else 1) == 0 or outer_it == num_outer_iter -1) :
             sparsity_mip = np.sum(sigma_current > 0.5)
             H_val_current = calculate_user_hamiltonian(A_cs_matrix, y_cs_vec, R_current, sigma_current, lambda_mip)
             history_H_orig.append(H_val_current)
             print(f"MIP Iter {outer_it:3d}: Sparsity = {sparsity_mip}, Hamiltonian_orig = {H_val_current:.4f}")

    if verbose: print("Alternating MIP Solver (Original Binary Sigma) finished.")
    return R_current, sigma_current, history_H_orig



import numpy as np
import itertools

def calculate_user_hamiltonian(A, y, R, sigma, lambda_mip):
    """
    Compute the original Hamiltonian:
      H = sum_{i,j} (A^T A)[i,j] * R[i]*R[j]*sigma[i]*sigma[j]
          - sum_i (A^T y)[i] * R[i]*sigma[i]
          + lambda_mip * sum_i sigma[i]
    """
    P = A.T @ A
    H_quadratic = float((P * np.outer(R * sigma, R * sigma)).sum())
    H_linear    = float((A.T @ y).dot(R * sigma))
    H_penalty   = lambda_mip * np.sum(sigma)
    return H_quadratic - H_linear + H_penalty

def solve_sigma_exact(A, y, R, lambda_mip):
    """
    Exact minimization of H(sigma) over sigma in {0,1}^N by brute force.
    """
    N = len(R)
    best_H = np.inf
    best_sigma = np.zeros(N, dtype=float)
    for bits in itertools.product((0.0, 1.0), repeat=N):
        sigma = np.array(bits)
        H = calculate_user_hamiltonian(A, y, R, sigma, lambda_mip)
        if H < best_H:
            best_H = H
            best_sigma = sigma
    return best_sigma

def solve_alternating_mip_B(A_cs_matrix,
                          y_cs_vec,
                          lambda_mip,
                          N_signal,
                          M_measure,
                          num_outer_iter=50,
                          num_R_iter=5,
                          num_sigma_iter=5,
                          initial_R=None,
                          initial_sigma=None,
                          verbose=True,
                          tol=1e-4,
                          sigma_method='icm'):
    """
    Alternating optimization with choice of sigma solver:
      sigma_method: 'icm' for ICM, 'exact' for brute-force QUBO.
    """
    # Initialization
    R_current     = np.copy(initial_R) if initial_R is not None else np.random.randn(N_signal)*0.1
    sigma_current = (np.copy(initial_sigma).astype(float)
                     if initial_sigma is not None else np.ones(N_signal))

    A_T       = A_cs_matrix.T
    P_matrix  = A_T @ A_cs_matrix
    A_T_y     = A_T @ y_cs_vec

    history_H = []
    H_prev    = calculate_user_hamiltonian(A_cs_matrix, y_cs_vec,
                                           R_current, sigma_current,
                                           lambda_mip)

    if verbose:
        print("Starting alternating MIP solver with CG for R, σ method:", sigma_method)

    for outer in range(num_outer_iter):
        # --- 1) Optimize sigma ---------------------------------------
        if sigma_method == 'icm':
            # iterative ICM
            for _ in range(num_sigma_iter):
                flips = 0
                v = R_current * sigma_current
                for i in range(N_signal):
                    total = P_matrix[i].dot(v)
                    off   = total - P_matrix[i,i] * v[i]
                    delta = (2*R_current[i]*off
                             + P_matrix[i,i]*R_current[i]**2
                             - R_current[i]*A_T_y[i]
                             + lambda_mip)
                    new_sigma = 1.0 if delta < 0 else 0.0
                    if new_sigma != sigma_current[i]:
                        sigma_current[i] = new_sigma
                        v[i] = R_current[i] * new_sigma
                        flips += 1
                if flips == 0:
                    break
        elif sigma_method == 'exact':
            sigma_current = solve_sigma_exact(A_cs_matrix, y_cs_vec,
                                              R_current, lambda_mip)
        else:
            raise ValueError("sigma_method must be 'icm' or 'exact'")

        # --- 2) Optimize R via Conjugate Gradient -------------------
        b_vec = sigma_current * A_T_y
        def matvec(x):
            return sigma_current * (P_matrix.dot(sigma_current * x))

        R = R_current.copy()
        r = b_vec - matvec(R)
        p = r.copy()
        rsold = r.dot(r)
        for _ in range(num_R_iter):
            Ap = matvec(p)
            alpha = rsold / (p.dot(Ap) + 1e-12)
            R += alpha * p
            r -= alpha * Ap
            rsnew = r.dot(r)
            if np.sqrt(rsnew) < 1e-6:
                break
            p = r + (rsnew/rsold)*p
            rsold = rsnew

        R_current = R * sigma_current

        # --- 3) Check convergence -----------------------------------
        H_curr = calculate_user_hamiltonian(A_cs_matrix, y_cs_vec,
                                            R_current, sigma_current,
                                            lambda_mip)
        history_H.append(H_curr)
        if verbose:
            print(f"Iter {outer+1:3d}: H = {H_curr:.6f}, ΔH = {H_curr - H_prev:.6f}, "
                  f"sparsity = {int(sigma_current.sum())}")
        if abs(H_curr - H_prev) < tol:
            if verbose:
                print("Converged (|ΔH| < tol).")
            break
        H_prev = H_curr

    if verbose:
        print("Finished alternating solver.")
    return R_current, sigma_current, history_H



def solve_matching_pursuit(A_cs_matrix,
                           y_cs_vec,
                           lambda_mip,
                           N_signal,
                           M_measure,
                           num_outer_iter=50,
                           num_R_iter=None,
                           num_sigma_iter=None,
                           initial_R=None,
                           initial_sigma=None,
                           verbose=True, tol=1e-5):
    """
    Matching Pursuit (MP) solver for L0‐regularized sparse recovery.
    Signature matches solve_alternating_mip; lambda_mip is used
    as a stopping threshold on max correlation.
    Returns: (R_current, sigma_current, history_H)
    """
    # Initialize
    R_current     = np.zeros(N_signal)
    sigma_current = np.zeros(N_signal)
    residual      = y_cs_vec.copy()
    history_H     = []

    if verbose:
        print("Starting Matching Pursuit solver")

    for it in range(num_outer_iter):
        # 1) compute correlations
        correlations = A_cs_matrix.T @ residual
        idx = int(np.argmax(np.abs(correlations)))
        max_corr = correlations[idx]

        # 2) stopping criterion
        if abs(max_corr) < tol:
            if verbose:
                print(f"MP stop at iter {it}: max|corr|={abs(max_corr):.4f} < lambda={lambda_mip}")
            break

        # 3) update coefficient and support
        atom_norm_sq = A_cs_matrix[:, idx].dot(A_cs_matrix[:, idx])
        R_current[idx] = max_corr / (atom_norm_sq + 1e-12)
        sigma_current[idx] = 1.0

        # 4) update residual
        residual -= A_cs_matrix[:, idx] * R_current[idx]

        # 5) record Hamiltonian
        H_curr = calculate_user_hamiltonian(A_cs_matrix, y_cs_vec,
                                            R_current, sigma_current,
                                            lambda_mip)
        history_H.append(H_curr)

        if verbose:
            print(f"MP iter {it+1}: picked atom {idx}, corr={max_corr:.4f}, H={H_curr:.6f}")

    if verbose:
        print("Finished Matching Pursuit solver")
    return R_current, sigma_current, history_H


def solve_omp(A_cs_matrix,
              y_cs_vec,
              lambda_mip,
              N_signal,
              M_measure,
              num_outer_iter=50,
              num_R_iter=None,
              num_sigma_iter=None,
              initial_R=None,
              initial_sigma=None,
              verbose=True,
              tol=1e-5):
    """
    Orthogonal Matching Pursuit (OMP) for L0-regularized sparse recovery.
    Signature matches solve_alternating_mip.
    
    Returns:
      R_current, sigma_current, history_H
    """
    # Initialize
    R_current     = np.zeros(N_signal)
    sigma_current = np.zeros(N_signal)
    residual      = y_cs_vec.copy()
    history_H     = []
    support       = []  # list of selected atom indices

    if verbose:
        print("Starting Orthogonal Matching Pursuit solver")

    for it in range(num_outer_iter):
        # 1) compute correlations
        correlations = A_cs_matrix.T @ residual
        idx = int(np.argmax(np.abs(correlations)))
        max_corr = correlations[idx]

        # 2) stopping criterion
        if abs(max_corr) < tol:
            if verbose:
                print(f"OMP stop at iter {it}: max|corr|={abs(max_corr):.4f} < lambda={lambda_mip}")
            break

        # 3) update support
        if idx not in support:
            support.append(idx)

        # 4) solve least squares on current support
        A_sub = A_cs_matrix[:, support]      # M_measure × |support|
        # solve A_sub * R_sub ≈ y_cs_vec
        R_sub, *_ = np.linalg.lstsq(A_sub, y_cs_vec, rcond=None)
        
        # 5) update R_current and sigma_current
        R_current[:]     = 0.0
        sigma_current[:] = 0.0
        for k, atom in enumerate(support):
            R_current[atom]     = R_sub[k]
            sigma_current[atom] = 1.0

        # 6) update residual
        residual = y_cs_vec - A_sub @ R_sub

        # 7) record Hamiltonian
        H_curr = calculate_user_hamiltonian(A_cs_matrix,
                                            y_cs_vec,
                                            R_current,
                                            sigma_current,
                                            lambda_mip)
        history_H.append(H_curr)
        if verbose:
            print(f"OMP iter {it+1}: support={support}, H={H_curr:.6f}")

    if verbose:
        print("Finished OMP solver")
    return R_current, sigma_current, history_H


def solve_cosamp(A_cs_matrix,
                 y_cs_vec,
                 lambda_mip,
                 target_sparsity,
                 N_signal,
                 M_measure,
                 num_outer_iter=50,
                 num_R_iter=None,
                 num_sigma_iter=None,
                 initial_R=None,
                 initial_sigma=None,
                 verbose=True):
    """
    Compressive Sampling Matching Pursuit (CoSaMP) solver for L0-regularized recovery.
    Signature extended with:
      target_sparsity : int, desired number of nonzeros in the solution

    Returns: R_current, sigma_current, history_H
    """
    s = max(1, int(target_sparsity))

    # Initialize
    residual      = y_cs_vec.copy()
    R_current     = np.zeros(N_signal)
    sigma_current = np.zeros(N_signal)
    support       = []
    history_H     = []

    if verbose:
        print(f"Starting CoSaMP with target sparsity s = {s}")

    for it in range(num_outer_iter):
        # 1) proxy correlation
        proxy = A_cs_matrix.T @ residual

        # 2) identify 2s largest
        omega = np.argsort(-np.abs(proxy))[:2*s]

        # 3) merge supports
        T = sorted(set(support) | set(omega.tolist()))

        # 4) least squares on T
        A_T = A_cs_matrix[:, T]
        a_T, *_ = np.linalg.lstsq(A_T, y_cs_vec, rcond=None)

        # 5) prune to s largest coefficients
        idx_sorted = np.argsort(-np.abs(a_T))[:s]
        new_support = [T[i] for i in idx_sorted]

        # 6) update signal estimate on new support
        A_S = A_cs_matrix[:, new_support]
        x_S, *_ = np.linalg.lstsq(A_S, y_cs_vec, rcond=None)

        # 7) update current R and sigma
        R_current[:]     = 0
        sigma_current[:] = 0
        for k, idx in enumerate(new_support):
            R_current[idx]     = x_S[k]
            sigma_current[idx] = 1

        # 8) update residual
        residual = y_cs_vec - A_S @ x_S

        # 9) record Hamiltonian
        H_val = calculate_user_hamiltonian(A_cs_matrix,
                                           y_cs_vec,
                                           R_current,
                                           sigma_current,
                                           lambda_mip)
        history_H.append(H_val)

        if verbose:
            res_norm = np.linalg.norm(residual)
            print(f"CoSaMP iter {it+1}: |res|={res_norm:.4e}, support={new_support}, H={H_val:.6f}")

        # 10) stopping criteria
        if np.linalg.norm(residual) < lambda_mip:
            if verbose:
                print("CoSaMP converged: residual norm below threshold")
            # break
        if set(new_support) == set(support):
            if verbose:
                print("CoSaMP converged: support unchanged")
            break

        support = new_support

    if verbose:
        print("Finished CoSaMP solver")
    return R_current, sigma_current, history_H
