"""
Tests for the penalties module.

This file contains tests for the penalty functions in the l0core.penalties module.
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from l0core.penalties import (
    penalty_terms_single_var,
    total_loss_function_single_var
)
from l0core.hamiltonians import hamiltonian_h_prime_single_var

class TestPenalties(unittest.TestCase):
    """Test class for penalty functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducibility
        np.random.seed(42)
        jax.config.update("jax_enable_x64", True)
        
        # Create small matrices and vectors for testing
        self.N = 10  # Number of coefficients
        self.M = 5   # Number of measurements
        
        # Generate a random sensing matrix
        self.A = np.random.normal(0, 1.0/np.sqrt(self.M), (self.M, self.N))
        self.A_T = self.A.T
        
        # Generate a sparse signal
        self.x_true = np.zeros(self.N)
        self.x_true[[0, 3, 7]] = [1.0, 2.0, -1.5]  # Make 3 entries non-zero
        
        # Generate measurements
        self.y = self.A @ self.x_true
        
        # Create R and binary variables for testing
        self.R = np.array([1.0, 0.5, 0.0, 2.0, 0.0, 0.0, 0.0, -1.5, 0.0, 0.0])
        
        # Create binary variables with values in [0,1]
        self.binary_perfect = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.binary_relaxed = np.array([0.9, 0.1, 0.3, 0.8, 0.2, 0.1, 0.0, 0.95, 0.15, 0.05])
        
        # Convert to JAX arrays
        self.A_jax = jnp.array(self.A_T)
        self.y_jax = jnp.array(self.y)
        self.R_jax = jnp.array(self.R)
        self.binary_perfect_jax = jnp.array(self.binary_perfect)
        self.binary_relaxed_jax = jnp.array(self.binary_relaxed)
        
        # Set the lambda parameter for regularization
        self.lambda_param = 0.1
        self.penalty_coeff = 1.0

    def test_penalty_terms_single_var_perfect_binary(self):
        """Test penalty terms with perfect binary values."""
        penalty = penalty_terms_single_var(self.binary_perfect_jax, self.penalty_coeff)
        
        # Perfect binary values (0 or 1) should give zero penalty
        self.assertAlmostEqual(float(penalty), 0.0, places=10)
    
    def test_penalty_terms_single_var_relaxed(self):
        """Test penalty terms with relaxed (non-binary) values."""
        penalty = penalty_terms_single_var(self.binary_relaxed_jax, self.penalty_coeff)
        
        # Calculate expected penalty: sum of (x_i * (1 - x_i))^2
        expected_penalty = 0.0
        for x in self.binary_relaxed:
            expected_penalty += (x * (1.0 - x))**2
        expected_penalty *= self.penalty_coeff
        
        # Convert JAX value to Python float for comparison
        penalty_value = float(penalty)
        
        self.assertAlmostEqual(penalty_value, expected_penalty, places=5)
    
    def test_penalty_scaling(self):
        """Test that penalty scales correctly with the coefficient."""
        penalty_1 = penalty_terms_single_var(self.binary_relaxed_jax, 1.0)
        penalty_2 = penalty_terms_single_var(self.binary_relaxed_jax, 2.0)
        
        # Doubling the coefficient should double the penalty
        self.assertAlmostEqual(float(penalty_2), float(penalty_1) * 2.0, places=5)
    
    def test_total_loss_function(self):
        """Test the total loss function (Hamiltonian + penalty)."""
        # Calculate Hamiltonian term
        h_prime = hamiltonian_h_prime_single_var(
            self.R_jax,
            self.binary_relaxed_jax,
            self.A_jax,
            self.y_jax,
            self.lambda_param
        )
        
        # Calculate penalty term
        penalty = penalty_terms_single_var(
            self.binary_relaxed_jax,
            self.penalty_coeff
        )
        
        # Calculate total loss
        total_loss = total_loss_function_single_var(
            self.R_jax,
            self.binary_relaxed_jax,
            self.A_jax,
            self.y_jax,
            self.lambda_param,
            self.penalty_coeff
        )
        
        # The total loss should be the sum of h_prime and penalty
        expected_total_loss = float(h_prime) + float(penalty)
        actual_total_loss = float(total_loss)
        
        self.assertAlmostEqual(actual_total_loss, expected_total_loss, places=5)

if __name__ == "__main__":
    unittest.main()
