import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from scipy.signal import find_peaks
from scipy.signal import convolve2d

# Simple Halftone (Thresholding)
def simple_halftone(image):
    height, width = image.shape
    threshold = np.mean(image)  # Calculate the average pixel value as the threshold
    halftoned_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            halftoned_image[y, x] = 255 if image[y, x] > threshold else 0
    return halftoned_image

# Advanced Halftone (Dithering method) - Optimized
def advanced_halftone(image):
    image = np.array(image, dtype=float)
    height, width = image.shape
    for y in range(height - 1):
        for x in range(1, width - 1):
            old_pixel = image[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            image[y, x] = new_pixel
            error = old_pixel - new_pixel
            image[y, x + 1] += error * 7 / 16
            image[y + 1, x - 1] += error * 3 / 16
            image[y + 1, x] += error * 5 / 16
            image[y + 1, x + 1] += error * 1 / 16
    return np.uint8(np.clip(image, 0, 255))
# Calculate histogram manually
def calculate_histogram(image):
    #Creates an array of size 256 initialized to zeros. Each index represents a pixel intensity (0–255).
    hist = np.zeros(256)
    for pixel in image.flatten(): #Iterates through each pixel value in the flattened 1D version of the image.
        hist[pixel] += 1
    return hist

# Histogram equalization using manual calculation
def histogram_equalization(image):
    hist = calculate_histogram(image)
    cdf = np.cumsum(hist) #Computes the cumulative sum of the histogram, resulting in the CDF. Each element in cdf represents the cumulative number of pixels up to that intensity.
    cdf_min = cdf[cdf > 0].min() ##Finds the smallest non-zero value in the CDF. This avoids issues with dark areas in the image.
    cdf_normalized = ((cdf - cdf_min) * 255) / (cdf[-1] - cdf_min) ##Subtracting cdf_min ensures the lower end starts at 0.
    # Dividing by (cdf[-1] - cdf_min) ensures the CDF is scaled to the full range.
    equalized_image = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            equalized_image[y, x] = cdf_normalized[image[y, x]]
    return np.uint8(equalized_image)

# Sobel operator
def apply_sobel(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    grad_x = np.zeros_like(image, dtype=float)
    grad_y = np.zeros_like(image, dtype=float)

    height, width = image.shape

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            region = image[y - 1:y + 2, x - 1:x + 2]
            grad_x[y, x] = np.sum(region * sobel_x)
            grad_y[y, x] = np.sum(region * sobel_y)

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return np.uint8(np.clip(magnitude, 0, 255))

# Prewitt operator
def apply_prewitt(image):
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    grad_x = np.zeros_like(image, dtype=float)
    grad_y = np.zeros_like(image, dtype=float)

    height, width = image.shape

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            region = image[y - 1:y + 2, x - 1:x + 2]
            grad_x[y, x] = np.sum(region * kernel_x)
            grad_y[y, x] = np.sum(region * kernel_y)

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return np.uint8(np.clip(magnitude, 0, 255))

# Kirsch operator
def apply_kirsch(image):
    # Convert to signed integers to handle subtraction safely
    image = image.astype(np.int16)

    # Define Kirsch masks for each direction
    kirsch_masks = [
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # North
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # South
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, 5]]),  # East
        np.array([[5, 5, -3], [0, 0, -3], [-3, -3, -3]]),  # West
        np.array([[-3, 5, 5], [-3, 0, 5], [5, -3, -3]]),  # Northeast
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),  # Northwest
        np.array([[5, -3, -3], [-3, 0, -3], [5, 5, -3]]),  # Southeast
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])   # Southwest
    ]

    gradient_images = []
    direction_names = ['North', 'South', 'East', 'West', 'Northeast', 'Northwest', 'Southeast', 'Southwest']

    max_gradient_value = 0
    max_gradient_direction = None
    max_gradient_image = None

    for mask in kirsch_masks:
        gradient = cv2.filter2D(image, -1, mask)
        # Normalize for better visualization (clipping before normalization)
        normalized_gradient = np.clip(gradient, 0, 255)
        normalized_gradient = cv2.normalize(normalized_gradient, None, 0, 255, cv2.NORM_MINMAX)
        gradient_images.append(normalized_gradient)
        # Print the gradient for this direction
        print(f"Gradient for {direction_names[len(gradient_images) - 1]} direction:")
        print(normalized_gradient)
        # Find the maximum gradient value and direction
        max_value = np.max(normalized_gradient)
        if max_value > max_gradient_value:
            max_gradient_value = max_value
            max_gradient_direction = direction_names[len(gradient_images) - 1]
            max_gradient_image = normalized_gradient

    # If no gradient images were processed or no valid direction found, return None
    if max_gradient_image is None or max_gradient_direction is None:
        print("No valid gradient direction found!")
        return None, None

    # Convert the image to uint8 before returning it
    max_gradient_image = np.uint8(np.clip(max_gradient_image, 0, 255))

    # Return the image and direction with the highest gradient
    return max_gradient_image, max_gradient_direction
#homogeneity
def homogeneity(the_image,threshold = 10):
    rows, cols = the_image.shape
    # Convert to signed integers to handle subtraction safely
    the_image = the_image.astype(np.int16)
    out_image = np.zeros_like(the_image, dtype=np.int16)
    # Compute homogeneity
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            max_diff = 0
            for a in range(-1, 2):
                for b in range(-1, 2):
                    # Safely calculate difference
                    diff = abs(the_image[i, j] - the_image[i + a, j + b])
                    if diff > max_diff:
                        max_diff = diff
            if max_diff >=threshold:
                out_image[i, j] = max_diff
    # Clip to valid range and return as uint8
    return np.uint8(np.clip(out_image,0,255))
#difference edge
def difference_edge(the_image):
    threshold =np.mean(the_image)
    rows, cols = the_image.shape
    out_image = np.zeros((rows, cols), dtype=np.int16)  # Intermediate output for max difference
    # Convert the image to int16 to handle negative differences
    image_int = the_image.astype(np.int16)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Pairs of pixels to compute differences
            pairs = [
                (image_int[i - 1, j], image_int[i + 1, j]),  # Vertical
                (image_int[i, j - 1], image_int[i, j + 1]),  # Horizontal
                (image_int[i - 1, j - 1], image_int[i + 1, j + 1]),  # Diagonal 1
                (image_int[i - 1, j + 1], image_int[i + 1, j - 1])  # Diagonal 2
            ]
            # Compute absolute differences for each pair
            differences = [abs(pair[0] - pair[1]) for pair in pairs]
            # Find the maximum difference
            max_diff = max(differences)
            # Assign the maximum difference to the output image
            out_image[i, j] = max_diff

    # Normalize the output image to 0-255 for better visualization
    normalized_out_image = (out_image - out_image.min()) * (255 / (out_image.max() - out_image.min()))
    normalized_out_image = normalized_out_image.astype(np.uint8)

    # Apply thresholding
    thresholded_image = np.where(normalized_out_image >= threshold, 255, 0).astype(np.uint8)

    return thresholded_image
'''
def gaus_7(the_image):
    rows, cols = the_image.shape
    out_image = np.zeros_like(the_image, dtype=np.int16)

    # Define the 7x7 and 9x9 masks
    mask_7x7 = np.array([
        [0, 0, -1, -1, -1, 0, 0],
        [0, -2, -3, -3, -3, -2, 0],
        [-1, -3, 5, 5, 5, -3, -1],
        [-1, -3, 5, 16, 5, -3, -1],
        [-1, -3, 5, 5, 5, -3, -1],
        [0, -2, -3, -3, -3, -2, 0],
        [0, 0, -1, -1, -1, 0, 0]
    ])
        conv_7x7 = convolve2d(the_image, mask_7x7, mode='same', boundary='fill', fillvalue=0)
# Normalize and scale the result for better visualization
        normalized_diff = (diff - diff.min()) * (255 / (diff.max() - diff.min()))
        out_image = normalized_diff.astype(np.uint8)
w
        return out_image '''
def gaus_d(the_image):
    # Define the 7x7 and 9x9 masks
    mask_7x7 = np.array([
        [0, 0, -1, -1, -1, 0, 0],
        [0, -2, -3, -3, -3, -2, 0],
        [-1, -3, 5, 5, 5, -3, -1],
        [-1, -3, 5, 16, 5, -3, -1],
        [-1, -3, 5, 5, 5, -3, -1],
        [0, -2, -3, -3, -3, -2, 0],
        [0, 0, -1, -1, -1, 0, 0]
    ])

    mask_9x9 = np.array([
        [0, 0, 0, -1, -1, -1, 0, 0, 0],
        [0, -2, -3, -3, -3, -3, -3, -2, 0],
        [0, -3, -2, -1, -1, -1, -2, -3, 0],
        [-1, -3, -1, 9, 9, 9, -1, -3, -1],
        [-1, -3, -1, 9, 19, 9, -1, -3, -1],
        [-1, -3, -1, 9, 9, 9, -1, -3, -1],
        [0, -3, -2, -1, -1, -1, -2, -3, 0],
        [0, -2, -3, -3, -3, -3, -3, -2, 0],
        [0, 0, 0, -1, -1, -1, 0, 0, 0]
    ])

    # Convolve the image with both masks
    conv_7x7 = convolve2d(the_image, mask_7x7, mode='same', boundary='fill', fillvalue=0)
    conv_9x9 = convolve2d(the_image, mask_9x9, mode='same', boundary='fill', fillvalue=0)

    # Calculate the absolute difference between the convolutions
    diff = np.abs( conv_7x7 - conv_9x9 )

    # Normalize and scale the result for better visualization
    normalized_diff = (diff - diff.min()) * (255 / (diff.max() - diff.min()))
    out_image = normalized_diff.astype(np.uint8)

    return np.uint8(np.clip(out_image, 0, 255))


def contrast_edge(image):
        rows, cols = image.shape
        out_image = np.zeros_like(image, dtype=np.float32)

        # Define the smoothing mask
        smoothing_mask = np.array([
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9]
        ])

        # Define the edge detection mask (EdgeDetectorMask)
        edge_mask = np.array([
            [-1, 0, -1],
            [0, 4, 0],
            [-1, 0, -1]
        ])

        # Perform edge detection and smoothing
        edge_output = cv2.filter2D(image, -1, edge_mask)
        average_output = cv2.filter2D(image, -1, smoothing_mask)
        average_output = average_output.astype(np.float64)

        average_output += 1e-10  # Add a small value to avoid division by zero

        # Calculate contrast edges
        contrast_edges = edge_output / average_output

        # Clip the output to [0, 255]
        contrast_edges = np.clip(contrast_edges, 0, 255)

        # Normalize to [0, 255]
        contrast_edges = cv2.normalize(contrast_edges, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to uint8 for display
        return np.uint8(contrast_edges)
def range_operator(the_image):
    rows, cols = the_image.shape
    out_image = np.zeros_like(the_image, dtype=np.int16)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = the_image[i - 1:i + 2, j - 1:j + 2]
            ranges =np.max(neighbors) - np.min(neighbors)
            out_image[i, j] = ranges
    return np.uint8(np.clip(out_image, 0, 255))

def variance_operator(the_image):
    rows, cols = the_image.shape
    out_image = np.zeros_like(the_image, dtype=np.int16)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = the_image[i - 1:i + 2, j - 1:j + 2]
            mean = np.mean(neighbors)
            variance =np.sum((neighbors - mean)**2)/9
            out_image[i, j] = min(max(variance, 0), 255)
    return np.uint8(np.clip(out_image, 0, 255))

# Low-pass filter (Smoothing filter)
def apply_low_pass(image):
        # New kernel as per the provided filter
        kernel = np.array([
            [0, 1/6, 0],
            [1/6 / 6, 2/6, 1/6],
            [0, 1/6, 0]
        ])
        # Apply the kernel to the image using convolution
        smoothed_image = cv2.filter2D(image, -1, kernel)
        return np.uint8(np.clip(smoothed_image, 0, 255))

# High-pass filter (Edge detection by the mask in the book)
def apply_high_pass(image):
    # First kernel provided by the user
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    # Apply the kernel to the image using convolution
    filtered_image = cv2.filter2D(image, -1, kernel)

    # Return the processed image, ensuring pixel values are valid
    return np.uint8(np.clip(filtered_image, 0, 255))

# median-pass filter (Edge detection by median-pass)
def apply_median_pass(image):
    height, width = image.shape
    filtered_image = np.zeros_like(image)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            neighborhood = image[i - 1:i + 2, j - 1:j + 2]
            median_value = np.median(neighborhood)
            filtered_image[i, j] = median_value

    return filtered_image
# Custom Image Operations
def ImageCopy(image1, image2):
    height, width = image1.shape
    # Convert images to a higher data type to avoid overflow
    image1 = image1.astype(np.int16)
    image2 = image2.astype(np.int16)

    added_image = np.zeros((height, width), dtype=np.int16)

    for i in range(height):
        for j in range(width):
            added_image[i, j] = image1[i, j] + image2[i, j]
            # Clamp to range [0, 255]
            added_image[i, j] = max(0, min(added_image[i, j], 255))

    # Convert back to uint8 for the final image
    return added_image.astype(np.uint8)


def subtract_image(image1, image2):
    height, width = image1.shape
    subtracted_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            subtracted_image[i, j] = image1[i, j] - image2[i, j]
            subtracted_image[i, j] = max(0, min(subtracted_image[i, j], 255))  # Clamp to 0-255

    return subtracted_image

def invert_image(image):
    height, width = image.shape
    inverted_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            inverted_image[i, j] = 255 - image[i, j]  # Invert pixel value

    return inverted_image


##########################MANUAL#################################
def manual_thresholding(image, value=255):
    # Apply preprocessing steps
    equalized = histogram_equalization(image)
    high_pass = apply_high_pass(equalized)
    smoothed = apply_low_pass(high_pass)

    # Input thresholds with validation
    while True:
        try:
            lower_thresh = int(input("Enter the lower threshold (0-255): "))
            upper_thresh = int(input("Enter the upper threshold (0-255): "))

            if 0 <= lower_thresh <= 255 and 0 <= upper_thresh <= 255:
                if lower_thresh <= upper_thresh:
                    break
                else:
                    print("Lower threshold must be less than or equal to the upper threshold.")
            else:
                print("Thresholds must be in the range 0-255.")
        except ValueError:
            print("Invalid input. Please enter integers only.")

    # Apply manual thresholding
    segmented_image = np.zeros_like(smoothed)
    segmented_image[(smoothed >= lower_thresh) & (smoothed <= upper_thresh)] = value

    return segmented_image


##########################PEAK#################################


# Main function for histogram peak threshold segmentation
def histogram_peak_threshold_segmentation(image):
    # Compute histogram
    hist = calculate_histogram(image).flatten()
    # Find peaks and calculate thresholds
    peaks_indices = find_histogram_peaks(hist)
    low_threshold, high_threshold = calculate_thresholds(peaks_indices, hist)

    print(f"Low Threshold: {low_threshold}, High Threshold: {high_threshold}")
    # Apply preprocessing steps
    equalized = histogram_equalization(image)
    high_pass = apply_high_pass(equalized)
    smoothed = apply_low_pass(high_pass)

    # Apply thresholding
    segmented_image = np.zeros_like(smoothed)
    segmented_image[(smoothed >= low_threshold) & (smoothed <= high_threshold)] = 255

    return segmented_image


# Function for finding histogram peaks
def find_histogram_peaks(hist, min_distance=10):
    peaks, _ = find_peaks(hist, height=0, distance=min_distance)  # Enforce a minimum distance between peaks
    sorted_peaks = sorted(peaks, key=lambda x: hist[x], reverse=True)  # Sort peaks by height
    return sorted_peaks[:2]  # Return top 2 peaks


# Function to calculate thresholds
def calculate_thresholds(peaks_indices, hist):
    if len(peaks_indices) < 2:
        # If less than two peaks are found, use a default threshold or a different strategy
        return np.min(hist), np.max(hist)

    # Calculate thresholds based on peak indices
    peak1 = peaks_indices[0]
    peak2 = peaks_indices[1]
    low_threshold = (peak1 + peak2) // 2
    high_threshold = peak2

    return low_threshold, high_threshold


##########################VALLEY#################################

# Main function to perform histogram-based segmentation
def histogram_valley_threshold_segmentation(image):
    # Apply preprocessing steps
    equalized = histogram_equalization(image)
    high_pass = apply_high_pass(equalized)
    smoothed = apply_low_pass(high_pass)

    # Calculate histogram
    hist_valley = calculate_histogram(image).flatten()

    # Find peaks and calculate thresholds
    peaks_indices = find_histogram_peaks(hist_valley)
    valley_point = find_valley_point(peaks_indices, hist_valley)
    low_threshold, high_threshold = calculate_valley_thresholds(peaks_indices, valley_point, hist_valley)

    print(f"Low Threshold: {low_threshold}, High Threshold: {high_threshold}")

    # Perform segmentation
    segmented_image = np.zeros_like(smoothed)
    segmented_image[(smoothed >= low_threshold) & (smoothed <= high_threshold)] = 255

    return segmented_image


# Function to find the valley point between two peaks
def find_valley_point(peaks_indices, hist_valley):
    if len(peaks_indices) < 2:
        raise ValueError("At least two peaks are required to find a valley point.")

    # Ensure peaks are sorted by their positions in the histogram
    peaks_indices = sorted(peaks_indices)

    # Get the range between the two most prominent peaks
    start, end = peaks_indices[0], peaks_indices[1]

    # Initialize the minimum value and valley point index
    min_valley = float('inf')
    valley_point = start

    # Iterate through the range between the peaks to find the valley
    for i in range(start, end + 1):
        if hist_valley[i] < min_valley:
            min_valley = hist_valley[i]
            valley_point = i

    return valley_point


# Function to calculate thresholds based on peaks and valley
def calculate_valley_thresholds(peaks_indices, valley_point, hist_valley):
    if len(peaks_indices) < 2:
        # If less than two peaks are found, use a default threshold or a different strategy
        return np.min(hist_valley), np.max(hist_valley)

    # Calculate thresholds based on valley and peak indices
    low_threshold = valley_point
    high_threshold = peaks_indices[1]
    return low_threshold, high_threshold


##########################Adaptive#################################

# Main function to perform adaptive histogram threshold segmentation
def adaptive_histogram_threshold(image):
    # Apply preprocessing steps
    equalized = histogram_equalization(image)
    high_pass = apply_high_pass(equalized)
    smoothed = apply_low_pass(high_pass)

    # Calculate histogram
    hist_ada = calculate_histogram(image).flatten()

    # Find peaks and calculate thresholds
    peaks_indices = find_histogram_peaks(hist_ada)
    valley_point = find_valley_point(peaks_indices, hist_ada)
    low_threshold, high_threshold = valley_high_low(peaks_indices, valley_point)

    print([low_threshold, high_threshold])

    # Perform segmentation
    segmented_image = np.zeros_like(smoothed)
    segmented_image[(smoothed >= low_threshold) & (smoothed <= high_threshold)] = 255

    # Calculate new thresholds based on mean values from segmented image
    background_mean, object_mean = calculate_means(segmented_image, smoothed)
    new_peaks_indices = [int(background_mean), int(object_mean)]
    new_low_threshold, new_high_threshold = valley_high_low(new_peaks_indices,
                                                            find_valley_point(new_peaks_indices, hist_ada))

    print([new_low_threshold, new_high_threshold])

    # Final segmentation
    final_segmented_image = np.zeros_like(smoothed)
    final_segmented_image[(smoothed >= new_low_threshold) & (smoothed <= new_high_threshold)] = 255

    return final_segmented_image


def valley_high_low(peaks_indices, valley_point):
    low_threshold = valley_point
    high_threshold = peaks_indices[-1] if len(peaks_indices) > 1 else valley_point
    return low_threshold, high_threshold


def calculate_means(segmented_image, original_image):
    object_pixels = original_image[segmented_image == 255]
    background_pixels = original_image[segmented_image == 0]

    object_mean = object_pixels.mean() if object_pixels.size > 0 else 0
    background_mean = background_pixels.mean() if background_pixels.size > 0 else 0

    return background_mean, object_mean
class EdgeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Edge Detection GUI")

        self.original_image = None
        self.grayscale_image = None
        self.display_image = None
        self.current_filter = None  # Track current filter
        self.selected_filters = set()  # Track selected filters

        # Create canvas for image display
        self.figure, self.ax = plt.subplots(1, 2, figsize=(10, 5))  # Two side-by-side plots
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")

        # Configure grid weights for resizing
        self.root.grid_rowconfigure(0, weight=1, minsize=300)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_columnconfigure(3, weight=1)

        # Buttons

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image, relief="raised")
        self.load_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        '''   
         self.gaus_7 = tk.Button(root, text="Load Image", command=self.gaus_7, relief="raised")
         self.gaus_7.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
         self.gaus_9 = tk.Button(root, text="Load Image", command=self.gaus_9, relief="raised")
         self.gaus_9.grid(row=1, column=2, padx=10, pady=10, sticky="ew")'''

        self.simple_halftone_button = tk.Button(root, text="Simple Halftone", command=self.apply_simple_halftone, relief="raised")
        self.simple_halftone_button.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        self.advanced_halftone_button = tk.Button(root, text="Advanced Halftone", command=self.apply_advanced_halftone, relief="raised")
        self.advanced_halftone_button.grid(row=1, column=2, padx=10, pady=10, sticky="ew")

        self.histogram_equalization_button = tk.Button(root, text="Histogram Equalization", command=self.apply_histogram_equalization, relief="raised")
        self.histogram_equalization_button.grid(row=1, column=3, padx=10, pady=10, sticky="ew")
        self.sobel_button = tk.Button(root, text="Sobel Edge Detection", command=self.apply_sobel, relief="raised")
        self.sobel_button.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        self.prewitt_button = tk.Button(root, text="Prewitt Edge Detection", command=self.apply_prewitt, relief="raised")
        self.prewitt_button.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        self.kirsch_button = tk.Button(root, text="Kirsch Edge Detection(N/A)", command=self.apply_kirsch, relief="raised")
        self.kirsch_button.grid(row=2, column=2, padx=10, pady=10, sticky="ew")
        self.homogeneity_button = tk.Button(root, text="Homogeneity Edge Detection", command=self.apply_homogeneity,
                                            relief="raised")
        self.homogeneity_button.grid(row=2, column=3, padx=10, pady=10, sticky="ew")

        self.difference_button = tk.Button(root, text="difference Edge Detection", command=self.apply_difference,
                                       relief="raised")
        self.difference_button.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        self.gaus_d = tk.Button(root, text="gaus difference", command=self.gaus_d, relief="raised")
        self.gaus_d.grid(row=3, column=1,padx=10, pady=10, sticky="ew")
        self.contrast_button = tk.Button(root, text="contrast Edge Detection", command=self.apply_contrast,
                                           relief="raised")
        self.contrast_button.grid(row=3, column=2, padx=10, pady=10, sticky="ew")

        self.range_button = tk.Button(root, text="range Edge Detection", command=self.apply_range,
                                         relief="raised")
        self.range_button.grid(row=3, column=3, padx=10, pady=10, sticky="ew")

        self.variance_button = tk.Button(root, text="variance Edge Detection", command=self.apply_variance,
                                      relief="raised")
        self.variance_button.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

        # Low-pass filter button
        self.low_pass_button = tk.Button(root, text="Low Pass Filter", command=self.apply_low_pass, relief="raised")
        self.low_pass_button.grid(row=4, column=1, padx=10, pady=10, sticky="ew")

        # High-pass filter button
        self.high_pass_button = tk.Button(root, text="High Pass Filter", command=self.apply_high_pass,  relief="raised")
        self.high_pass_button.grid(row=4, column=3, padx=10, pady=10, sticky="ew")

        # median-pass filter button
        self.median_pass_button = tk.Button(root, text="Median Pass Filter", command=self.apply_median_pass, relief="raised")
        self.median_pass_button.grid(row=4, column=2, padx=10, pady=10, sticky="ew")

        # ImageCopy filter button
        self.ImageCopy_button = tk.Button(root, text="ImageCopy Filter", command=self.apply_ImageCopy,
                                            relief="raised")
        self.ImageCopy_button.grid(row=5, column=0, padx=10, pady=10, sticky="ew")

        # subtract_image filter button
        self.subtract_image_button = tk.Button(root, text="subtract image Filter", command=self.apply_subtract_image, relief="raised")
        self.subtract_image_button.grid(row=5, column=1, padx=10, pady=10, sticky="ew")

        # invert filter button
        self.invert_button = tk.Button(root, text="invert Filter", command=self.apply_invert,
                                          relief="raised")
        self.invert_button.grid(row=5, column=2, padx=10, pady=10, sticky="ew")
        self.histo_manual_button = tk.Button(root, text="histogram manual Image", command=self.apply_ManualHisto_image,
                                           relief="raised")
        self.histo_manual_button.grid(row=6, column=0, padx=10, pady=10, sticky="ew")

        # histo peak image button
        self.histo_peak_button = tk.Button(root, text="histogram peak Image", command=self.apply_PeakHisto_image, relief="raised")
        self.histo_peak_button.grid(row=6, column=1, padx=10, pady=10, sticky="ew")

        # histo valley image button
        self.histo_valley_button = tk.Button(root, text="histogram valley Image", command=self.apply_ValleyHisto_image,
                                           relief="raised")
        self.histo_valley_button.grid(row=6, column=2, padx=10, pady=10, sticky="ew")

        # histo adaptive image button
        self.histo_adaptive_button = tk.Button(root, text="histogram adaptive Image", command=self.apply_AdaptiveHisto_image,
                                            relief="raised")
        self.histo_adaptive_button.grid(row=6, column=3, padx=10, pady=10, sticky="ew")

        # Clear image button
        self.clear_button = tk.Button(root, text="Clear Image", command=self.clear_image, relief="raised")
        self.clear_button.grid(row=7, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

        # Current filter label
        self.filter_label = tk.Label(root, text="Current Filter Applied: None", anchor="w")
        self.filter_label.grid(row=8, column=0, columnspan=4, padx=10, pady=10, sticky="w")
        self.threshold_label = tk.Label(root, text="Threshold: N/A", relief="raised")
        self.threshold_label.grid(row=9, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

        self.update_button_colors()

    def update_button_colors(self):
        # Set default colors for buttons
        buttons = [
           #self.gaus_7,self.gaus_9,
            self.gaus_d, self.simple_halftone_button,self.variance_button,self.contrast_button,self.range_button,self.difference_button, self.homogeneity_button, self.advanced_halftone_button, self.histogram_equalization_button,
            self.sobel_button, self.prewitt_button, self.kirsch_button, self.low_pass_button, self.median_pass_button, self.high_pass_button,self.ImageCopy_button,self.invert_button,self.subtract_image_button,self.histo_manual_button, self.histo_peak_button,self.histo_valley_button,self.histo_adaptive_button]
        for button in buttons:
            button.config(bg="#d3d3d3", fg="black")

        # Set color for selected filter buttons (blue)
        for button in self.selected_filters:
            button.config(bg="#4682b4", fg="white")

        # Set color for applied filter button (green)
        if self.current_filter:
            self.current_filter.config(bg="#32cd32", fg="white")

    def update_canvas(self, original_image, display_image):
        self.ax[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        self.ax[0].set_title("Original Image")
        self.ax[0].axis("off")
        self.ax[1].imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        self.ax[1].set_title("Processed Image")
        self.ax[1].axis("off")
        self.canvas.draw()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            image = cv2.imread(file_path)
            self.original_image = image
            ##The result will be a 2D array with shape (height, width) representing the grayscale image.
            #This line of code converts a color image to grayscale by applying a weighted sum of its Red, Green, and Blue (RGB) channels.
            self.grayscale_image = np.uint8(0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2])
            #self.grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.display_image = self.grayscale_image
            self.update_canvas(self.original_image, self.display_image)
            self.filter_label.config(text="Current Filter Applied: None")
            self.selected_filters = set()  # Reset selected filters
            self.current_filter = None
            self.update_button_colors()

    def clear_image(self):
        if self.grayscale_image is not None:
            self.display_image = self.grayscale_image.copy()
            self.filter_label.config(text="Current Filter Applied: None")
            self.current_filter = None
            self.selected_filters = set()  # Reset selected filters
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)

    def apply_simple_halftone(self):
        if self.grayscale_image is not None:
            self.display_image = simple_halftone(self.grayscale_image)
            self.threshold_label.config(text=f"Threshold: {np.mean(self.grayscale_image):.2f}")
            self.current_filter = self.simple_halftone_button
            self.filter_label.config(text="Current Filter Applied: Simple Halftone")
            self.selected_filters.add(self.simple_halftone_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)

    def apply_advanced_halftone(self):
        if self.grayscale_image is not None:
            self.display_image = advanced_halftone(self.grayscale_image)
            self.current_filter = self.advanced_halftone_button
            self.filter_label.config(text="Current Filter Applied: Advanced Halftone")
            self.selected_filters.add(self.advanced_halftone_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)

    def apply_histogram_equalization(self):
        if self.grayscale_image is not None:
            self.display_image = histogram_equalization(self.grayscale_image)
            self.current_filter = self.histogram_equalization_button
            self.filter_label.config(text="Current Filter Applied: Histogram Equalization")
            self.selected_filters.add(self.histogram_equalization_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)


    def apply_sobel(self):
        if self.grayscale_image is not None:
            self.display_image = apply_sobel(self.grayscale_image)
            self.current_filter = self.sobel_button
            self.filter_label.config(text="Current Filter Applied: Sobel Edge Detection")
            self.selected_filters.add(self.sobel_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)

    def apply_prewitt(self):
        if self.grayscale_image is not None:
            self.display_image = apply_prewitt(self.grayscale_image)
            self.current_filter = self.prewitt_button
            self.filter_label.config(text="Current Filter Applied: Prewitt Edge Detection")
            self.selected_filters.add(self.prewitt_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)

    def apply_kirsch(self):
        if self.grayscale_image is not None:
            best_direction_image, best_direction = apply_kirsch(self.grayscale_image)
            self.display_image = best_direction_image
            self.kirsch_button.config(text=f"Kirsch Edge Detection ({best_direction} used)")
            self.current_filter = self.kirsch_button
            self.filter_label.config(text="Current Filter Applied: Kirsch Edge Detection")
            self.selected_filters.add(self.kirsch_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)
    def apply_homogeneity(self):
        if self.grayscale_image is not None:
            self.display_image = homogeneity(self.grayscale_image)
            self.current_filter = self.homogeneity_button
            self.filter_label.config(text="Current Filter Applied: homogeneity Edge Detection")
            self.selected_filters.add(self.homogeneity_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)
    def apply_range(self):
        if self.grayscale_image is not None:
            self.display_image = range_operator(self.grayscale_image)
            self.current_filter = self.range_button
            self.filter_label.config(text="Current Filter Applied: Range Edge Detection")
            self.selected_filters.add(self.range_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)

    def apply_difference(self):
        if self.grayscale_image is not None:
            self.display_image = difference_edge(self.grayscale_image)
            self.current_filter = self.difference_button
            self.filter_label.config(text="Current Filter Applied: difference Edge Detection")
            self.selected_filters.add(self.difference_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)

    def apply_contrast(self):
        if self.grayscale_image is not None:
            self.display_image = contrast_edge(self.grayscale_image)
            self.current_filter = self.contrast_button
            self.filter_label.config(text="Current Filter Applied: contrast Edge Detection")
            self.selected_filters.add(self.contrast_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)

    def apply_variance(self):
        if self.grayscale_image is not None:
            self.display_image = variance_operator(self.grayscale_image)
            self.current_filter = self.variance_button
            self.filter_label.config(text="Current Filter Applied: variance Edge Detection")
            self.selected_filters.add(self.variance_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)

    def apply_low_pass(self):
        if self.grayscale_image is not None:
            self.display_image = apply_low_pass(self.grayscale_image)
            self.current_filter = self.low_pass_button
            self.filter_label.config(text="Current Filter Applied: Low Pass Filter")
            self.selected_filters.add(self.low_pass_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)

    def apply_high_pass(self):
        if self.grayscale_image is not None:
            self.display_image = apply_high_pass(self.grayscale_image)
            self.current_filter = self.high_pass_button
            self.filter_label.config(text="Current Filter Applied: High Pass Filter")
            self.selected_filters.add(self.high_pass_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)
    def apply_median_pass(self):
        if self.grayscale_image is not None:
            self.display_image = apply_median_pass(self.grayscale_image)
            self.current_filter = self.median_pass_button
            self.filter_label.config(text="Current Filter Applied: Median Pass Filter")
            self.selected_filters.add(self.median_pass_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)

    def apply_ImageCopy(self):
        if self.grayscale_image is not None:
            self.display_image = ImageCopy(self.grayscale_image, self.grayscale_image)
            self.current_filter = self.ImageCopy_button
            self.filter_label.config(text="Current Filter Applied: Image copy Filter")
            self.selected_filters.add(self.ImageCopy_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)

    def gaus_d(self):
        if self.grayscale_image is not None:
            self.display_image = gaus_d(self.grayscale_image)
            self.current_filter = self.gaus_d
            self.filter_label.config(text="Current Filter Applied: gaus difference Filter")
            self.selected_filters.add(self.gaus_d)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)
    def apply_subtract_image(self):
        if self.grayscale_image is not None:
            self.display_image = subtract_image(self.grayscale_image, self.grayscale_image)
            self.current_filter = self.subtract_image_button
            self.filter_label.config(text="Current Filter Applied: subtract Filter")
            self.selected_filters.add(self.subtract_image_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)

    def apply_invert(self):
        if self.grayscale_image is not None:
            self.display_image = invert_image(self.grayscale_image)
            self.current_filter = self.invert_button
            self.filter_label.config(text="Current Filter Applied: invert Filter")
            self.selected_filters.add(self.invert_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)

    def apply_ManualHisto_image(self):
        if self.grayscale_image is not None:
            self.display_image = manual_thresholding(self.grayscale_image)
            self.current_filter = self.histo_manual_button
            self.filter_label.config(text="Current Filter Applied: histogram manual threshold")
            self.selected_filters.add(self.histo_manual_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)
    def apply_PeakHisto_image(self):
        if self.grayscale_image is not None:
            self.display_image = histogram_peak_threshold_segmentation(self.grayscale_image)
            self.current_filter = self.histo_peak_button
            self.filter_label.config(text="Current Filter Applied: histogram peak threshold")
            self.selected_filters.add(self.histo_peak_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)
    def apply_ValleyHisto_image(self):
        if self.grayscale_image is not None:
            self.display_image = histogram_valley_threshold_segmentation(self.grayscale_image)
            self.current_filter = self.histo_valley_button
            self.filter_label.config(text="Current Filter Applied: histogram Valley threshold")
            self.selected_filters.add(self.histo_valley_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)
    def apply_AdaptiveHisto_image(self):
        if self.grayscale_image is not None:
            self.display_image = adaptive_histogram_threshold(self.grayscale_image)
            self.current_filter = self.histo_adaptive_button
            self.filter_label.config(text="Current Filter Applied: histogram adaptive threshold")
            self.selected_filters.add(self.histo_adaptive_button)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)

'''
      def gaus_7(self):
        if self.grayscale_image is not None:
            self.display_image = gaus_7(self.grayscale_image)
            self.current_filter = self.gaus_7
            self.filter_label.config(text="Current Filter Applied: gaus7x7 Filter")
            self.selected_filters.add(self.gaus_7)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image)

    def gaus_9(self):
        if self.grayscale_image is not None:
            self.display_image = gaus_9(self.grayscale_image)
            self.current_filter = self.gaus_9
            self.filter_label.config(text="Current Filter Applied: gaus9x9 Filter")
            self.selected_filters.add(self.gaus_9)
            self.update_button_colors()
            self.update_canvas(self.original_image, self.display_image) '''

# Create and run the app
root = tk.Tk()
app = EdgeDetectionApp(root)
root.mainloop()
