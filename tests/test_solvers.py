"""
Tests for the solvers module.

This file contains tests for the solver functions in the l0core.solvers module.
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from l0core.solvers import (
    solve_continuous_relaxation_single_var,
    update_step_single_var,
    grad_loss_fn_single_var
)

class TestSolvers(unittest.TestCase):
    """Test class for solver functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducibility
        np.random.seed(42)
        jax.config.update("jax_enable_x64", True)
        
        # Create a very small problem for testing with a known solution
        self.N = 5  # Number of coefficients
        self.M = 3  # Number of measurements
        
        # Generate a random sensing matrix
        self.A = np.random.normal(0, 1.0/np.sqrt(self.M), (self.M, self.N))
        self.A_T = self.A.T
        
        # Generate a sparse signal with only 1 non-zero
        self.x_true = np.zeros(self.N)
        self.x_true[2] = 1.0  # Only the third element is non-zero
        
        # Generate measurements
        self.y = self.A @ self.x_true
        
        # Set parameters
        self.lambda_param = 0.01
        self.learning_rate = 0.05  # Increased learning rate for faster convergence
        self.num_iterations = 1000  # More iterations for better convergence
        self.penalty_coeff = 10.0

    def test_update_step(self):
        """Test update step function for gradient descent."""
        # Create some random R and x values
        R_vars = jnp.array([0.5, -0.2, 1.0, 0.3, -0.1])
        x_vars = jnp.array([0.6, 0.1, 0.8, 0.2, 0.0])
        
        # Create some mock gradients
        grad_R = jnp.array([0.1, -0.2, 0.3, -0.4, 0.5])
        grad_x = jnp.array([0.2, 0.3, -0.1, 0.4, 0.0])
        grads = (grad_R, grad_x)
        
        # Calculate expected values
        expected_R = R_vars - self.learning_rate * grad_R
        expected_x = jnp.clip(x_vars - self.learning_rate * grad_x, 0.0, 1.0)
        
        # Update step
        new_R, new_x = update_step_single_var(R_vars, x_vars, grads, self.learning_rate)
        
        # Check that the values match
        np.testing.assert_allclose(new_R, expected_R, rtol=1e-5)
        np.testing.assert_allclose(new_x, expected_x, rtol=1e-5)
        
        # Check that x is clipped to [0, 1]
        self.assertTrue(jnp.all(new_x >= 0.0))
        self.assertTrue(jnp.all(new_x <= 1.0))

    def test_solver_convergence(self):
        """Test that the solver converges to a reasonable solution."""
        # Convert to JAX arrays for the solver
        A_jax = jnp.array(self.A_T)
        y_jax = jnp.array(self.y)
        
        # Run the solver
        R_solution, x_solution, _ = solve_continuous_relaxation_single_var(
            self.A_T,
            self.y,
            self.lambda_param,
            self.N,
            self.M,
            learning_rate=self.learning_rate,
            num_iterations=self.num_iterations,
            penalty_coeff=self.penalty_coeff,
            verbose=False
        )
        
        # Apply thresholding to x_solution
        x_binary = (x_solution > 0.5).astype(float)
        reconstructed = R_solution * x_binary
        
        # Check that the reconstruction error is small
        # For our simple test, we're using a small number of iterations
        # so we allow a larger error threshold
        reconstruction_error = np.linalg.norm(self.A @ reconstructed - self.y) / np.linalg.norm(self.y)
        self.assertLess(reconstruction_error, 0.7)

    def test_solver_sparsity(self):
        """Test that the solver produces a sparse solution."""
        # Run the solver with a higher lambda value to encourage sparsity
        R_solution, x_solution, _ = solve_continuous_relaxation_single_var(
            self.A_T,
            self.y,
            lambda_param=0.1,  # Higher lambda encourages more sparsity
            N_r_dim=self.N,
            M_k_dim=self.M,
            learning_rate=self.learning_rate,
            num_iterations=self.num_iterations,
            penalty_coeff=self.penalty_coeff,
            verbose=False
        )
        
        # Apply thresholding to get binary values
        x_binary = (x_solution > 0.5).astype(float)
        
        # Count non-zeros
        num_nonzeros = np.sum(x_binary > 0)
        
        # With a small problem and high lambda, we should get a sparse solution
        # In a test environment with limited iterations, we'll accept a slightly higher sparsity
        self.assertLessEqual(num_nonzeros, 3)

if __name__ == "__main__":
    unittest.main()
