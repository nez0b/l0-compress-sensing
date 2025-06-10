"""
Example script demonstrating basic usage of the l0core package for L0 optimization.

This script shows how to:
1. Generate a simple sparse signal
2. Create a sensing matrix and measurements
3. Solve the L0 optimization problem using continuous relaxation
4. Compare the recovered signal with the original

Usage:
    python basic_example.py
"""

import numpy as np
import matplotlib.pyplot as plt

from l0core.solvers import solve_continuous_relaxation_single_var, solve_continuous_relaxation_sgd


def main():
    """Run the example."""
    # Generate a sparse signal
    n = 100  # signal dimension
    k = 10   # sparsity
    
    print("Generating a sparse signal with dimension", n, "and", k, "non-zeros")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create sparse signal with k non-zeros
    x_true = np.zeros(n)
    non_zero_indices = np.random.choice(n, k, replace=False)
    x_true[non_zero_indices] = np.random.normal(0, 1, k)
    
    # Create a sensing matrix and measurements
    m = n // 2  # compressed number of measurements
    A = np.random.normal(0, 1/np.sqrt(m), (m, n))
    y = A @ x_true
    
    print(f"Created measurements of dimension {m} (compression ratio: {n/m:.1f}x)")
    
    # Solve using L0 optimization with continuous relaxation
    print("Solving with L0 optimization...")
    lambda_param = 0.25
    
    # Call the solver
    R_solution, x_solution, loss_history = solve_continuous_relaxation_sgd(
        A.T,
        y,
        lambda_param,
        n,
        m,
        learning_rate=0.001,
        num_iterations=5000,
        penalty_coeff=1.,
        verbose=True
    )
    
    # Apply thresholding to get binary values
    x_binary = (x_solution > 0.5).astype(float)
    recovered_signal = R_solution * x_binary
    
    # Calculate recovery error
    error = np.linalg.norm(recovered_signal - x_true) / np.linalg.norm(x_true)
    recovered_sparsity = np.sum(np.abs(recovered_signal) > 1e-6)
    
    print("\nResults:")
    print(f"Original sparsity: {k}")
    print(f"Recovered sparsity: {recovered_sparsity}")
    print(f"Relative error: {error:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    plt.subplot(3, 1, 1)
    plt.stem(x_true, markerfmt='bo', linefmt='b-', basefmt='b-')
    plt.title('Original Signal')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.stem(recovered_signal, markerfmt='ro', linefmt='r-', basefmt='r-')
    plt.title(f'Recovered Signal (Error: {error:.4f})')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.stem(np.abs(x_true - recovered_signal), markerfmt='go', linefmt='g-', basefmt='g-')
    plt.title('Absolute Error')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
