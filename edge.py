import numpy as np
import cv2
import matplotlib.pyplot as plt

def convol(image, kernel):
    # Get the dimensions of the image and kernel
    image_h, image_w = image.shape
    kernel_size = kernel.shape[0]
    print (kernel.shape[0])
    kernel_radius = kernel_size // 2
    
    # Initialize the result matrix
    result = np.zeros_like(image)
    
    # Pad the image to handle borders
    padded_image = np.pad(image, ((kernel_radius, kernel_radius), (kernel_radius, kernel_radius)), mode='constant')
    
    # Perform convolution
    for y in range(kernel_radius, image_h + kernel_radius):
        for x in range(kernel_radius, image_w + kernel_radius):
            # Extract the region of interest (ROI)
            roi = padded_image[y - kernel_radius:y + kernel_radius + 1, x - kernel_radius:x + kernel_radius + 1]
            # Compute the convolution sum
            result[y - kernel_radius, x - kernel_radius] = np.sum(roi * kernel)
    
    return result

def sobel(image, threshold=100):
    # Convert image to grayscale
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_scale = cv2.GaussianBlur(gray_scale,(3,3),0)
    
    # Define Sobel kernels for horizontal and vertical edge detection
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    # Convolve the image with Sobel kernels
    grad_x = convol(gray_scale, sobel_x)
    grad_y = convol(gray_scale, sobel_y)
    
    # Compute the gradient magnitude
    grad_mag = np.sqrt(np.square(grad_x) + np.square(grad_y))
    
    # Normalize gradient magnitude to [0, 255]
    grad_mag *= 255.0 / grad_mag.max()
    grad_mag[grad_mag < threshold] = 0

    return grad_mag
    

# Load the image
image = cv2.imread('Photos/table.png')

# Resize the image to improve processing speed
image = cv2.resize(image, (500, 500))
ima_blur = cv2.GaussianBlur(image,(3,3),0)

# Perform edge detection using Sobel operator with custom convolution
edges = sobel(ima_blur, threshold=150)

# Save the detected edges as a PNG image
cv2.imwrite('edge.png', edges)


# Display the original and edge-detected images using Matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edges Detected')
plt.axis('off')

plt.show()
