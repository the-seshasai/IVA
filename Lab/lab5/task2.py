import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


import math

# Create a Gaussian kernel
def gaussian_kernel(size, sigma=1):
    kernel = np.zeros((size, size))
    mean = size // 2
    sum_val = 0
    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-((i - mean)**2 + (j - mean)**2) / (2 * sigma**2))
            sum_val += kernel[i, j]
    return kernel / sum_val  # Normalize the kernel

# Manually apply convolution with the Gaussian kernel
def apply_convolution(image, kernel):
    image_height, image_width = image.shape
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
    blurred_image = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            blurred_image[i, j] = np.sum(padded_image[i:i + kernel_size, j:j + kernel_size] * kernel)
    
    return blurred_image


image = load_image('orange.jpg')  
# Apply Gaussian blur with a 5x5 kernel and sigma = 1
gaussian_kernel_5x5 = gaussian_kernel(5, sigma=1)
blurred_image = apply_convolution(grayscale_image, gaussian_kernel_5x5)
show_image(blurred_image, title="Blurred Image")


