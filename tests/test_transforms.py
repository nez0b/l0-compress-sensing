"""
Tests for the transforms module.

This file contains tests for the transform functions in the l0core.transforms module.
"""

import unittest
import numpy as np
from l0core.transforms import (
    haar2_transform_flat,
    inverse_haar2_transform_from_flat,
    dct2_transform_flat,
    inverse_dct2_transform_flat
)

class TestTransforms(unittest.TestCase):
    """Test class for transform functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create simple test images
        self.test_image_8x8 = np.ones((8, 8))
        # Add some pattern to make it less uniform
        self.test_image_8x8[2:6, 2:6] = 2.0
        self.test_image_8x8[4:7, 4:7] = 0.5
        
        # Create a random test image
        np.random.seed(42)
        self.random_image_16x16 = np.random.rand(16, 16)

    def test_haar_transform_forward_inverse(self):
        """Test that Haar wavelet transform is invertible."""
        # Apply forward transform
        coeffs_flat, coeffs_shape, coeff_slices = haar2_transform_flat(self.test_image_8x8)
        
        # Apply inverse transform
        reconstructed = inverse_haar2_transform_from_flat(coeffs_flat, coeffs_shape, coeff_slices)
        
        # Check that the reconstructed image matches the original
        np.testing.assert_allclose(self.test_image_8x8, reconstructed, rtol=1e-10)
    
    def test_haar_transform_random_image(self):
        """Test Haar transform on a random image."""
        # Apply forward transform
        coeffs_flat, coeffs_shape, coeff_slices = haar2_transform_flat(self.random_image_16x16)
        
        # Apply inverse transform
        reconstructed = inverse_haar2_transform_from_flat(coeffs_flat, coeffs_shape, coeff_slices)
        
        # Check that the reconstructed image matches the original
        np.testing.assert_allclose(self.random_image_16x16, reconstructed, rtol=1e-10)
    
    def test_dct_transform_forward_inverse(self):
        """Test that DCT transform is invertible."""
        # Apply forward transform
        coeffs_flat, shape = dct2_transform_flat(self.test_image_8x8)
        
        # Apply inverse transform
        reconstructed = inverse_dct2_transform_flat(coeffs_flat, shape)
        
        # Check that the reconstructed image matches the original
        np.testing.assert_allclose(self.test_image_8x8, reconstructed, rtol=1e-10)
    
    def test_dct_transform_random_image(self):
        """Test DCT transform on a random image."""
        # Apply forward transform
        coeffs_flat, shape = dct2_transform_flat(self.random_image_16x16)
        
        # Apply inverse transform
        reconstructed = inverse_dct2_transform_flat(coeffs_flat, shape)
        
        # Check that the reconstructed image matches the original
        np.testing.assert_allclose(self.random_image_16x16, reconstructed, rtol=1e-10)

    def test_haar_transform_shapes(self):
        """Test that shapes are preserved correctly in Haar transform."""
        # Apply forward transform
        coeffs_flat, coeffs_shape, coeff_slices = haar2_transform_flat(self.test_image_8x8)
        
        # Check that the total number of coefficients matches the image size
        self.assertEqual(len(coeffs_flat), self.test_image_8x8.size)
        
    def test_dct_transform_shapes(self):
        """Test that shapes are preserved correctly in DCT transform."""
        # Apply forward transform
        coeffs_flat, shape = dct2_transform_flat(self.test_image_8x8)
        
        # Check that the total number of coefficients matches the image size
        self.assertEqual(len(coeffs_flat), self.test_image_8x8.size)
        
        # Check that the shape is preserved
        self.assertEqual(shape, self.test_image_8x8.shape)

if __name__ == "__main__":
    unittest.main()
