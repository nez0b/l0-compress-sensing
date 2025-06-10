# L0opt: L0 Optimization for Compressed Sensing

A Python package implementing L0 optimization methods for compressed sensing applications.

## Overview

This package provides a modular implementation of various L0 optimization algorithms for compressed sensing, focusing on efficient recovery of sparse signals. The implementation includes:

- JAX-based continuous relaxation solvers
- Alternating minimization methods
- QUBO-based solvers with quantum computing integration
- Various transform utilities (Haar wavelet, DCT)

## Package Structure

The `l0core` package is organized into the following modules:

- `transforms.py`: Transform utilities for converting between image and coefficient domains (Haar wavelet and DCT)
- `hamiltonians.py`: Different Hamiltonian formulations used in L0 optimization
- `penalties.py`: Penalty functions for enforcing binary constraints
- `solvers.py`: Core JAX and alternating minimization solvers
- `qubo_solvers.py`: QUBO-based and hybrid solvers (with QCI Dirac3 integration)

## Installation

```bash
# Clone the repository
git clone https://github.com/username/L0opt.git
cd L0opt

# Install the package and dependencies
pip install -e .
```

## Requirements

- Python >= 3.10
- JAX
- NumPy
- SciPy
- PyWavelets
- Matplotlib
- scikit-image
- eqc-models (optional, for QUBO solvers)

## Transform Functions

The package provides several transform utilities:

### Haar Wavelet Transform

```python
from l0core.transforms import haar2_transform_flat, inverse_haar2_transform_from_flat

# Forward transform
coeffs_flat, coeffs_shape, coeff_slices = haar2_transform_flat(image, level=1)

# Inverse transform
reconstructed = inverse_haar2_transform_from_flat(coeffs_flat, coeffs_shape, coeff_slices)
```

### DCT Transform

```python
from l0core.transforms import dct2_transform_flat, inverse_dct2_transform_flat

# Forward transform
coeffs_flat, image_shape = dct2_transform_flat(image)

# Inverse transform
reconstructed = inverse_dct2_transform_flat(coeffs_flat, image_shape)
```

## Usage Examples

### Basic Sparse Signal Recovery

```python
import numpy as np
from l0core.solvers import solve_continuous_relaxation_single_var

# Generate a sparse signal
n = 100  # signal dimension
k = 10   # sparsity (number of non-zeros)
x_true = np.zeros(n)
x_true[np.random.choice(n, k, replace=False)] = np.random.normal(0, 1, k)

# Create a sensing matrix and measurements
m = n // 4  # compressed number of measurements
A = np.random.normal(0, 1/np.sqrt(m), (m, n))
y = A @ x_true

# Solve using L0 optimization with continuous relaxation
lambda_param = 0.01
R_solution, x_solution = solve_continuous_relaxation_single_var(
    A.T,
    y,
    lambda_param,
    n,
    m,
    learning_rate=0.01,
    num_iterations=2000,
    penalty_coeff=10.0
)

# Apply thresholding to get binary values
recovered_signal = R_solution * (x_solution > 0.5)
```

### Image Processing with Wavelets

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from l0core.transforms import haar2_transform_flat, inverse_haar2_transform_from_flat
from l0core.solvers import solve_continuous_relaxation_single_var

# Load a test image
image = data.camera()

# Apply Haar wavelet transform
coeffs_flat, coeffs_shape, coeff_slices = haar2_transform_flat(image)

# Create sensing matrix and measurements
n = len(coeffs_flat)
m = n // 4
A = np.random.normal(0, 1/np.sqrt(m), (m, n))
y = A @ coeffs_flat

# Solve using L0 optimization
lambda_param = 0.01
R_solution, x_solution = solve_continuous_relaxation_single_var(
    A.T, y, lambda_param, n, m
)

# Recover the image
recovered_coeffs = R_solution * (x_solution > 0.5)
recovered_image = inverse_haar2_transform_from_flat(
    recovered_coeffs, coeffs_shape, coeff_slices
)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(recovered_image, cmap='gray')
plt.title('Recovered Image')
plt.show()
```

## Examples and Notebooks

The package includes example scripts and notebooks demonstrating usage:

- `examples/basic_example.py`: Simple sparse signal recovery
- `examples/using_l0core.ipynb`: Comprehensive demonstration of package features
- `L0opt.ipynb`: The original notebook with detailed explanations

## Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific test
python -m unittest tests.test_transforms
```

## Citation

If you find this package useful in your research, please consider citing:

```bibtex
@software{l0opt2023,
  author = {Wang, Po-Jen},
  title = {L0opt: L0 Optimization for Compressed Sensing},
  year = {2023},
  url = {https://github.com/username/L0opt}
}
```

## License

[MIT License](LICENSE)