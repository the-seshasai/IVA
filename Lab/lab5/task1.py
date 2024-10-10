import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    return image

def convert_to_grayscale(image):
    if len(image.shape) == 3:  # If image is RGB
        grayscale_image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        grayscale_image = image  # If already grayscale
    return grayscale_image

def show_image(image, title="Image"):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')  # Hide axes
    plt.show()

# Gaussian kernel for blurring
def gaussian_kernel(size, sigma=1):
    kernel = np.zeros((size, size))
    mean = size // 2
    sum_val = 0
    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-((i - mean)**2 + (j - mean)**2) / (2 * sigma**2))
            sum_val += kernel[i, j]
    return kernel / sum_val  # Normalize the kernel

# Convolution operation
def apply_convolution(image, kernel):
    image_height, image_width = image.shape
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
    result_image = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            result_image[i, j] = np.sum(padded_image[i:i + kernel_size, j:j + kernel_size] * kernel)
    
    return result_image

# Histogram equalization to enhance contrast
def histogram_equalization(image):
    image_flat = image.flatten()
    histogram = np.zeros(256)

    # Calculate histogram
    for pixel in image_flat:
        histogram[int(pixel)] += 1

    # Calculate cumulative distribution function (CDF)
    cdf = np.cumsum(histogram)
    cdf_normalized = cdf / cdf[-1]  # Normalize CDF

    # Map pixel values using CDF
    equalized_image = np.interp(image_flat, range(0, 256), cdf_normalized * 255)
    return equalized_image.reshape(image.shape).astype(np.uint8)

# Sobel kernels for edge detection
sobel_x_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

sobel_y_kernel = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]])

# Sobel edge detection
def sobel_edge_detection(image):
    sobel_x = apply_convolution(image, sobel_x_kernel)
    sobel_y = apply_convolution(image, sobel_y_kernel)
    
    sobel_magnitude = np.hypot(sobel_x, sobel_y)
    
    sobel_magnitude = np.clip(sobel_magnitude, 0, 255)
    sobel_magnitude = (sobel_magnitude / np.max(sobel_magnitude)) * 255  


    threshold = 50
    sobel_magnitude[sobel_magnitude < threshold] = 0
    
    return sobel_magnitude.astype(np.uint8)


image_path = 'orange.jpg'
image = load_image(image_path)  
grayscale_image = convert_to_grayscale(image)
show_image(grayscale_image, title="Grayscale Image")

# Apply contrast enhancement
equalized_image = histogram_equalization(grayscale_image)

# Apply Gaussian blur to reduce noise
gaussian_kernel_5x5 = gaussian_kernel(5, sigma=1)
blurred_image = apply_convolution(equalized_image, gaussian_kernel_5x5)

# Apply Sobel edge detection
edges = sobel_edge_detection(blurred_image)
show_image(edges, title="Sobel Edge Detection with Threshold")
