"""
Hamiltonian functions for L0 optimization.

This module provides different Hamiltonian formulations used in L0 optimization
for compressed sensing, including both the original user-defined Hamiltonian and
its various relaxed or modified versions.
"""

import numpy as np
import jax
import jax.numpy as jnp

def hamiltonian_h_prime_single_var(R_vars, x_vars, A_input_matrix, y_input_data_vec, lambda_param):
    """
    Calculates the Hamiltonian using only x_vars as binary decision variables.
    
    Uses the relaxed form of the Hamiltonian where s_r = R_r * x_r:
    H' = 0.5 * s^T @ (A^T @ A) @ s - (A^T @ y)^T @ s + lambda * sum(x_r)
    
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
        Vector y_data, expected to be y_cs (measurement vector).
        Shape (M_k_dim,)
    lambda_param : float
        Regularization parameter lambda

    Returns:
    --------
    float : The value of the Hamiltonian
    """
    # s_r = R_r * x_r (element-wise product)
    s_vector = R_vars * x_vars # Shape (N_r_dim,)

    # Q_matrix = A_input_matrix @ A_input_matrix.T
    Q_matrix = A_input_matrix @ A_input_matrix.T  # Shape (N_r_dim, N_r_dim)
    
    # Term 1 (Quadratic): 0.5 * s_vector.T @ Q_matrix @ s_vector
    term1_quadratic = 0.5 * jnp.dot(s_vector, Q_matrix @ s_vector)

    # Term 2 (Linear): - (A_input_matrix @ y_input_data_vec).T @ s_vector
    q_vector_equivalent = A_input_matrix @ y_input_data_vec # Shape (N_r_dim,)
    term2_linear = -jnp.dot(q_vector_equivalent, s_vector)

    # Term 3 (L0-like penalty): lambda * sum_r x_r
    term3_l0 = lambda_param * jnp.sum(x_vars)

    return term1_quadratic + term2_linear + term3_l0

def hamiltonian_h_prime_old(R_vars, x_vars, A_input_matrix, y_input_data_vec, lambda_param):
    """
    Calculates the Hamiltonian H' as defined in the original formulation.
    H' = sum_{r<r'} sum_k A_rk A_r'k R_r R_r' x_r x_r'
         - sum_r sum_k y_datak A_rk R_r x_r
         + lambda * sum_r x_r

    Parameters:
    -----------
    R_vars : jnp.ndarray
        Vector R of shape (N_r_dim,)
    x_vars : jnp.ndarray
        Vector x of shape (N_r_dim,)
    A_input_matrix : jnp.ndarray
        Matrix A of shape (N_r_dim, M_k_dim)
        In CS context, this will be A_cs.T
    y_input_data_vec : jnp.ndarray
        Vector y_data (y^k) of shape (M_k_dim,)
        In CS context, this will be y_cs
    lambda_param : float
        Regularization parameter lambda

    Returns:
    --------
    float : The value of H'
    """
    N_r_dim = R_vars.shape[0]

    # V_r = R_r * x_r
    V = R_vars * x_vars # Shape (N_r_dim,)

    # Term 1: sum_{r<r'} sum_k A_rk A_r'k R_r R_r' x_r x_r'
    # sum_k A_rk A_r'k is (A_input_matrix @ A_input_matrix.T)_{r,r'}
    Q = A_input_matrix @ A_input_matrix.T  # Shape (N_r_dim, N_r_dim)
    
    # Outer product V_r * V_r'
    V_outer_V = V[:, None] * V[None, :] # Shape (N_r_dim, N_r_dim)
    
    term1_matrix = Q * V_outer_V
    # Sum over r < r' (upper triangle, excluding diagonal)
    term1 = jnp.sum(jnp.triu(term1_matrix, k=1))

    # Term 2: - sum_r sum_k y_datak A_rk R_r x_r
    # sum_k y_datak A_rk is (A_input_matrix @ y_input_data_vec)_r
    A_y = A_input_matrix @ y_input_data_vec # Shape (N_r_dim,)
    term2 = -jnp.sum(A_y * V)

    # Term 3: lambda * sum_r x_r
    term3 = lambda_param * jnp.sum(x_vars)

    h_prime = term1 + term2 + term3
    return h_prime

def hamiltonian_h_prime(R_vars, x_vars, A_input_matrix, y_input_data_vec, lambda_param):
    """
    Calculates a modified Hamiltonian based on the user's formulation,
    where the quadratic term now includes diagonal components, resulting in
    0.5 * s.T @ (A_input_matrix @ A_input_matrix.T) @ s,
    which corresponds to 0.5 * s.T @ (A_cs.T @ A_cs) @ s for the CS problem.
    
    The full objective calculated (excluding penalties and constants) is:
    0.5 * s.T @ P_cs @ s - q_cs.T @ s + lambda * sum(x_r)
    where s_r = R_r * x_r, P_cs = A_cs.T @ A_cs, q_cs = A_cs.T @ y_cs.

    Parameters:
    -----------
    R_vars : jnp.ndarray
        Vector R of shape (N_r_dim,)
    x_vars : jnp.ndarray
        Vector x of shape (N_r_dim,)
    A_input_matrix : jnp.ndarray
        Matrix A, expected to be A_cs.T (transpose of sensing matrix)
        Shape (N_r_dim, M_k_dim)
    y_input_data_vec : jnp.ndarray
        Vector y_data, expected to be y_cs (measurement vector)
        Shape (M_k_dim,)
    lambda_param : float
        Regularization parameter lambda

    Returns:
    --------
    float : The value of the modified Hamiltonian
    """
    # s_r = R_r * x_r (element-wise product)
    s_vector = R_vars * x_vars # Shape (N_r_dim,)

    # Q_matrix = A_input_matrix @ A_input_matrix.T
    # If A_input_matrix is A_cs.T, then Q_matrix = A_cs.T @ A_cs (= P_cs)
    Q_matrix = A_input_matrix @ A_input_matrix.T  # Shape (N_r_dim, N_r_dim)
    
    # Term 1 (Quadratic): 0.5 * s_vector.T @ Q_matrix @ s_vector
    # This now includes the diagonal terms, making it positive semi-definite.
    term1_quadratic = 0.5 * jnp.dot(s_vector, Q_matrix @ s_vector)

    # Term 2 (Linear): - (A_input_matrix @ y_input_data_vec).T @ s_vector
    # If A_input_matrix is A_cs.T and y_input_data_vec is y_cs,
    # then A_input_matrix @ y_input_data_vec = A_cs.T @ y_cs (= q_cs)
    q_vector_equivalent = A_input_matrix @ y_input_data_vec # Shape (N_r_dim,)
    term2_linear = -jnp.dot(q_vector_equivalent, s_vector)

    # Term 3 (L0-like penalty): lambda * sum_r x_r
    term3_l0 = lambda_param * jnp.sum(x_vars)

    modified_h_prime = term1_quadratic + term2_linear + term3_l0
    return modified_h_prime

def calculate_user_hamiltonian(A_cs_matrix, y_cs_vec, R_vec, sigma_vec_binary, lambda_val):
    """
    Calculates the user's original Hamiltonian H for the Mixed Integer Programming (MIP) formulation.
    
    H = sum_{r<r'} sum_k A_r^k A_r'^k R_r R_r' sigma_r sigma_r'
        - sum_r sum_k y^k A_r^k R_r sigma_r
        + lambda * sum_r sigma_r
    
    Parameters:
    -----------
    A_cs_matrix : numpy.ndarray
        Sensing matrix of shape (M_meas, N_sig)
    y_cs_vec : numpy.ndarray
        Measurement vector of shape (M_meas,)
    R_vec : numpy.ndarray
        Vector R of shape (N_sig,)
    sigma_vec_binary : numpy.ndarray
        Binary vector sigma of shape (N_sig,)
    lambda_val : float
        Regularization parameter lambda
    
    Returns:
    --------
    float : The value of the Hamiltonian H
    """
    N_sig = R_vec.shape[0]
    # A_H is A_cs_matrix.T, shape (N_sig, M_meas)
    A_H = A_cs_matrix.T 
    y_H = y_cs_vec  # Shape (M_meas,)
     
    term1 = 0.0
    # sum_{r<r'} sum_k A_r^k A_r'^k R_r R_r' sigma_r sigma_r'
    for r in range(N_sig):
        if sigma_vec_binary[r] < 0.5 or abs(R_vec[r]) < 1e-9:
            continue
        for rp in range(r + 1, N_sig):  # r < r'
            if sigma_vec_binary[rp] < 0.5 or abs(R_vec[rp]) < 1e-9:
                continue
            # sum_k A_r^k A_rp^k = dot product of r-th row of A_H and rp-th row of A_H
            sum_k_A_rk_A_rpk = np.dot(A_H[r, :], A_H[rp, :])
            term1 += sum_k_A_rk_A_rpk * R_vec[r] * R_vec[rp]  # sigma_r and sigma_rp are 1
             
    term2 = 0.0
    # - sum_r sum_k y^k A_r^k R_r sigma_r
    for r in range(N_sig):
        if sigma_vec_binary[r] < 0.5 or abs(R_vec[r]) < 1e-9:
            continue
        # sum_k y^k A_r^k = dot product of y_H and r-th row of A_H
        sum_k_yk_A_rk = np.dot(y_H, A_H[r, :])
        term2 -= sum_k_yk_A_rk * R_vec[r]  # sigma_r is 1
         
    term3 = lambda_val * np.sum(sigma_vec_binary > 0.5)
    return term1 + term2 + term3

def calculate_h_prime_relaxed(A_cs_matrix, y_cs_vec, R_vec, x_vec, lambda_h_prime):
    """
    Calculates the relaxed Hamiltonian H' based on the user's formulation:
    H' = sum_{r<r'} sum_k A_rk A_r'k R_r R_r' x_r x_r'
         - sum_r sum_k y_datak A_rk R_r x_r
         + lambda * sum_r x_r
    where A_rk is (A_cs_matrix.T)[r,k]
    
    Parameters:
    -----------
    A_cs_matrix : numpy.ndarray
        Sensing matrix of shape (M, N)
    y_cs_vec : numpy.ndarray
        Measurement vector of shape (M,)
    R_vec : numpy.ndarray
        Vector R of shape (N,)
    x_vec : numpy.ndarray
        Vector x of shape (N,) with values in [0,1]
    lambda_h_prime : float
        Regularization parameter lambda
        
    Returns:
    --------
    float : The value of the relaxed Hamiltonian
    """
    N_signal = R_vec.shape[0]
    s_vec = R_vec * x_vec

    # P_matrix = A_cs.T @ A_cs
    P_matrix = A_cs_matrix.T @ A_cs_matrix
    # q_vector = A_cs.T @ y_cs
    q_vector = A_cs_matrix.T @ y_cs_vec

    term1 = 0.0
    # sum_{r<r'} P_rr' s_r s_r'
    for r in range(N_signal):
        for rp in range(r + 1, N_signal):  # r < r'
            term1 += P_matrix[r, rp] * s_vec[r] * s_vec[rp]
            
    term2 = 0.0
    # - sum_r q_r s_r
    term2 = -np.dot(q_vector, s_vec)
        
    term3 = lambda_h_prime * np.sum(x_vec)
    
    return term1 + term2 + term3
