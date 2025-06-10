"""
Penalty functions for L0 optimization.

This module provides penalty functions used in various L0 optimization approaches
to enforce constraints on binary variables.
"""

import jax.numpy as jnp

def penalty_terms_single_var(x_vars, penalty_coeff):
    """
    Calculates penalty term for binary constraint using only x:
    x_r(1-x_r) = 0, which encourages x_r to be either 0 or 1
    
    Parameters:
    -----------
    x_vars : jnp.ndarray
        Vector x of shape (N,) with values in [0,1]
    penalty_coeff : float
        Weight coefficient for the penalty term
        
    Returns:
    --------
    float : The value of the penalty
    """
    # x(1-x) = 0 constraint
    binary_constraint = x_vars * (1.0 - x_vars)
    penalty = penalty_coeff * jnp.sum(binary_constraint**2)
    
    return penalty

def penalty_terms_original(x_vars, y_internal_vars, penalty_coeff1, penalty_coeff2):
    """
    Calculates penalty terms for the constraints:
    1. x_r + y_r = 1  => penalty_coeff1 * sum_r (x_r + y_r - 1)^2
    2. x_r * y_r = 0  => penalty_coeff2 * sum_r (x_r * y_r)^2
    
    Parameters:
    -----------
    x_vars : jnp.ndarray
        Vector x of shape (N,) with values in [0,1]
    y_internal_vars : jnp.ndarray
        Vector y of shape (N,) with values in [0,1]
    penalty_coeff1 : float
        Weight coefficient for the x_r + y_r = 1 constraint
    penalty_coeff2 : float
        Weight coefficient for the x_r * y_r = 0 constraint
        
    Returns:
    --------
    float : The sum of both penalty terms
    """
    constraint1_residuals = x_vars + y_internal_vars - 1.0
    penalty1 = penalty_coeff1 * jnp.sum(constraint1_residuals**2)

    constraint2_residuals = x_vars * y_internal_vars
    penalty2 = penalty_coeff2 * jnp.sum(constraint2_residuals**2)
    
    return penalty1 + penalty2

def total_loss_function_single_var(R_vars, x_vars, A_input_matrix, 
                                   y_input_data_vec, lambda_param, 
                                   penalty_coeff):
    """
    Calculates the total loss: H' + binary constraint penalty for single variable formulation.
    
    Imports the hamiltonian here to avoid circular imports.
    
    Parameters:
    -----------
    R_vars : jnp.ndarray
        Vector R of shape (N_r_dim,)
    x_vars : jnp.ndarray
        Vector x of shape (N_r_dim,) - relaxed binary variables
    A_input_matrix : jnp.ndarray
        Matrix A, expected to be A_cs.T (transpose of sensing matrix)
        Shape (N_r_dim, M_k_dim)
    y_input_data_vec : jnp.ndarray
        Vector y_data, expected to be y_cs (measurement vector)
        Shape (M_k_dim,)
    lambda_param : float
        Regularization parameter lambda
    penalty_coeff : float
        Weight coefficient for the binary constraint penalty
        
    Returns:
    --------
    float : The total loss value (H' + penalty)
    """
    from l0core.hamiltonians import hamiltonian_h_prime_single_var
    
    h_prime_val = hamiltonian_h_prime_single_var(R_vars, x_vars, A_input_matrix, 
                                              y_input_data_vec, lambda_param)
    penalties_val = penalty_terms_single_var(x_vars, penalty_coeff)
    return h_prime_val + penalties_val

def total_loss_function_original_hamiltonian(R_vars, x_vars, y_internal_vars, 
                                             A_input_matrix, y_input_data_vec, lambda_param, 
                                             penalty_coeff1, penalty_coeff2):
    """
    Calculates the total loss: H' + penalties for the original formulation.
    
    Imports the hamiltonian here to avoid circular imports.
    
    Parameters:
    -----------
    R_vars : jnp.ndarray
        Vector R of shape (N_r_dim,)
    x_vars : jnp.ndarray
        Vector x of shape (N_r_dim,)
    y_internal_vars : jnp.ndarray
        Vector y of shape (N_r_dim,)
    A_input_matrix : jnp.ndarray
        Matrix A of shape (N_r_dim, M_k_dim)
        In CS context, this will be A_cs.T
    y_input_data_vec : jnp.ndarray
        Vector y_data (y^k) of shape (M_k_dim,)
        In CS context, this will be y_cs
    lambda_param : float
        Regularization parameter lambda
    penalty_coeff1 : float
        Weight coefficient for the x_r + y_r = 1 constraint
    penalty_coeff2 : float
        Weight coefficient for the x_r * y_r = 0 constraint
        
    Returns:
    --------
    float : The total loss value (H' + penalties)
    """
    from l0core.hamiltonians import hamiltonian_h_prime
    
    h_prime_val = hamiltonian_h_prime(R_vars, x_vars, A_input_matrix, y_input_data_vec, lambda_param)
    penalties_val = penalty_terms_original(x_vars, y_internal_vars, penalty_coeff1, penalty_coeff2)
    return h_prime_val + penalties_val
