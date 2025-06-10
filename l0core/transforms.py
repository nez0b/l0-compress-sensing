"""
Transform utilities for L0 compressed sensing.

This module provides functions for transforming between image and coefficient domains,
including Haar wavelet and DCT transforms.
"""

import numpy as np
import pywt
from scipy.fftpack import dct, idct

def haar2_transform_flat(image, level=1):
    """
    Apply 2D Haar wavelet transform and flatten the coefficients.
    
    Parameters:
    -----------
    image : ndarray
        Input 2D image
    level : int, optional
        Decomposition level for the wavelet transform
        
    Returns:
    --------
    flat_coeffs : ndarray
        Flattened wavelet coefficients
    coeffs_arr_shape : tuple
        Shape of the coefficient array before flattening
    coeff_slices : list
        Slices information needed for reconstruction
    """
    coeffs = pywt.wavedec2(image, 'haar', level=level)
    coeffs_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    return coeffs_arr.flatten(), coeffs_arr.shape, coeff_slices

def ihaar2_transform_flat(flat_coeffs, coeffs_arr_shape, coeff_slices, target_image_shape=None):
    """
    Reconstruct an image from flattened Haar wavelet coefficients.
    
    Parameters:
    -----------
    flat_coeffs : ndarray
        Flattened wavelet coefficients
    coeffs_arr_shape : tuple
        Shape of the coefficient array before flattening
    coeff_slices : list
        Slices information needed for reconstruction
    target_image_shape : tuple, optional
        Target shape of the reconstructed image. If None, the reconstructed image shape is used as is.
        
    Returns:
    --------
    reconstructed_image : ndarray
        Reconstructed 2D image
    """
    coeffs_arr = flat_coeffs.reshape(coeffs_arr_shape)
    coeffs_list = pywt.array_to_coeffs(coeffs_arr, coeff_slices, output_format='wavedec2')
    reconstructed_image = pywt.waverec2(coeffs_list, 'haar')
    
    # Ensure reconstructed image has the requested shape by truncating if necessary
    if target_image_shape is not None:
        h, w = target_image_shape
        return reconstructed_image[:h, :w]
    
    return reconstructed_image

def inverse_haar2_transform_from_flat(flat_coeffs, coeffs_arr_shape, coeff_slices, target_image_shape=None):
    """
    Reconstruct an image from flattened Haar wavelet coefficients.
    Alias for ihaar2_transform_flat for API consistency.
    
    Parameters:
    -----------
    flat_coeffs : ndarray
        Flattened wavelet coefficients
    coeffs_arr_shape : tuple
        Shape of the coefficient array before flattening
    coeff_slices : list
        Slices information needed for reconstruction
    target_image_shape : tuple, optional
        Target shape of the reconstructed image. If None, the reconstructed image shape is used as is.
        
    Returns:
    --------
    reconstructed_image : ndarray
        Reconstructed 2D image
    """
    return ihaar2_transform_flat(flat_coeffs, coeffs_arr_shape, coeff_slices, target_image_shape)

def dct2(block):
    """
    Apply 2D DCT transform.
    
    Parameters:
    -----------
    block : ndarray
        Input 2D image/block
    
    Returns:
    --------
    dct_coeffs : ndarray
        2D DCT coefficients
    """
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """
    Apply inverse 2D DCT transform.
    
    Parameters:
    -----------
    block : ndarray
        2D DCT coefficients
    
    Returns:
    --------
    image : ndarray
        Reconstructed 2D image/block
    """
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def dct2_transform_flat(image):
    """
    Apply 2D DCT transform and flatten the coefficients.
    
    Parameters:
    -----------
    image : ndarray
        Input 2D image
        
    Returns:
    --------
    flat_coeffs : ndarray
        Flattened DCT coefficients
    image_shape : tuple
        Shape of the original image (needed for reconstruction)
    """
    return dct2(image).flatten(), image.shape

def idct2_transform_flat(flat_coeffs, image_shape):
    """
    Reconstruct an image from flattened DCT coefficients.
    
    Parameters:
    -----------
    flat_coeffs : ndarray
        Flattened DCT coefficients
    image_shape : tuple
        Shape of the target image
        
    Returns:
    --------
    reconstructed_image : ndarray
        Reconstructed 2D image
    """
    return idct2(flat_coeffs.reshape(image_shape))

def inverse_dct2_transform_flat(flat_coeffs, image_shape):
    """
    Reconstruct an image from flattened DCT coefficients.
    Alias for idct2_transform_flat for API consistency.
    
    Parameters:
    -----------
    flat_coeffs : ndarray
        Flattened DCT coefficients
    image_shape : tuple
        Shape of the target image
        
    Returns:
    --------
    reconstructed_image : ndarray
        Reconstructed 2D image
    """
    return idct2_transform_flat(flat_coeffs, image_shape)
