

import numpy as np
import cv2

def binarise(image, threshold=128):
    # Converting image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarising image using the given threshold
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    
    return binary_image

def hough(image, theta_res=1, rho_res=1, threshold=100):
    # Defining the range of theta (0 to 180 degrees) and rho (-max_distance to max_distance)
    theta_range = np.deg2rad(np.arange(0, 180, theta_res))
    max_distance = int(np.sqrt(image.shape[0]**2 + image.shape[1]**2))
    rho_range = np.arange(-max_distance, max_distance + 1, rho_res)
    
    # Initializing the Hough accumulator
    accumulator = np.zeros((len(rho_range), len(theta_range)), dtype=np.uint64)
    
    # Finding edge pixels in the image
    edge_pixels = np.argwhere(image > 0)
    
    # Loop over each edge pixel
    for y, x in edge_pixels:
        # Loop over each theta value
        for theta_idx, theta in enumerate(theta_range):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            rho_idx = np.argmin(np.abs(rho_range - rho))
            accumulator[rho_idx, theta_idx] += 1
    
    # Finding indices of accumulator cells above the threshold
    rho_idxs, theta_idxs = np.where(accumulator >= threshold)
    
    # Extracting rho and theta values for detected lines
    rhos = rho_range[rho_idxs]
    thetas = np.rad2deg(theta_range[theta_idxs])
    
    return rhos, thetas

def lines(image, rhos, thetas):
    for rho, theta in zip(rhos, thetas):
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Reading the edge image
edge_image = cv2.imread('edge.png')


# Binarising the edge image
binary_edge_image = binarise(edge_image, threshold=231)  # Adjustable threshold

# Applying Hough Transform
rhos, thetas = hough(binary_edge_image, theta_res=1, rho_res=1, threshold=101)  # Adjustable threshold

# Reading original image
original_image = cv2.imread('Photos/table.png')
original_image = cv2.resize(original_image, (500, 500))


# Drawing detected lines on the original image
lines_image = original_image.copy()
lines(lines_image, rhos, thetas)

# Saving the final result
cv2.imwrite('detected_lines.png', lines_image)

# Displaying the final result
cv2.imshow('Detected Lines', lines_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
