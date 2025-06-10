"""
Tests for the hamiltonians module.

This file contains tests for the Hamiltonian functions in the l0core.hamiltonians module.
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from l0core.hamiltonians import hamiltonian_h_prime_single_var

class TestHamiltonians(unittest.TestCase):
    """Test class for Hamiltonian functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create small matrices and vectors for testing
        self.N = 5  # Number of coefficients
        self.M = 3  # Number of measurements
        
        # Generate a random sensing matrix
        self.A = np.random.normal(0, 1.0/np.sqrt(self.M), (self.M, self.N))
        self.A_T = self.A.T
        
        # Create R and binary variables for testing
        self.R = np.array([1.0, 0.5, 0.0, -1.0, 2.0])
        self.binary = np.array([1.0, 0.0, 0.0, 1.0, 1.0])
        
        # Generate a signal based on R and binary
        self.signal = self.R * self.binary  # [1.0, 0.0, 0.0, -1.0, 2.0]
        
        # Generate measurements
        self.y = self.A @ self.signal
        
        # Convert to JAX arrays
        self.A_jax = jnp.array(self.A_T)
        self.y_jax = jnp.array(self.y)
        self.R_jax = jnp.array(self.R)
        self.binary_jax = jnp.array(self.binary)
        
        # Set the lambda parameter for regularization
        self.lambda_param = 0.1

    def test_hamiltonian_h_prime_single_var(self):
        """Test the hamiltonian_h_prime_single_var function."""
        # Calculate the Hamiltonian using the function
        H = hamiltonian_h_prime_single_var(
            self.R_jax, 
            self.binary_jax, 
            self.A_jax, 
            self.y_jax, 
            self.lambda_param
        )
        
        # Calculate the expected value manually
        s = self.signal
        AtA = self.A.T @ self.A
        Aty = self.A.T @ self.y
        
        expected_H = 0.5 * s @ AtA @ s - Aty @ s + self.lambda_param * np.sum(self.binary)
        
        # Convert JAX result to numpy for comparison
        H_np = np.array(H)
        
        # Test that the values are close
        self.assertAlmostEqual(H_np, expected_H, places=5)

    def test_hamiltonian_with_zero_binary(self):
        """Test the hamiltonian with all binary variables set to zero."""
        # Set all binary variables to zero
        zero_binary = jnp.zeros_like(self.binary_jax)
        
        # Calculate the Hamiltonian
        H = hamiltonian_h_prime_single_var(
            self.R_jax, 
            zero_binary, 
            self.A_jax, 
            self.y_jax, 
            self.lambda_param
        )
        
        # Since all binary variables are zero, the Hamiltonian should be zero
        self.assertAlmostEqual(float(H), 0.0, places=10)

    def test_hamiltonian_with_lambda_scaling(self):
        """Test that lambda properly scales the sparsity term."""
        # Calculate with lambda = 0.1
        H1 = hamiltonian_h_prime_single_var(
            self.R_jax, 
            self.binary_jax, 
            self.A_jax, 
            self.y_jax, 
            0.1
        )
        
        # Calculate with lambda = 0.2
        H2 = hamiltonian_h_prime_single_var(
            self.R_jax, 
            self.binary_jax, 
            self.A_jax, 
            self.y_jax, 
            0.2
        )
        
        # The difference should be exactly lambda_diff * sum(binary)
        lambda_diff = 0.1
        expected_diff = lambda_diff * np.sum(self.binary)
        actual_diff = float(H2) - float(H1)
        
        self.assertAlmostEqual(actual_diff, expected_diff, places=5)

if __name__ == "__main__":
    unittest.main()
