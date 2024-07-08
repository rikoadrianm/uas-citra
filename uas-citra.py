import numpy as np
import matplotlib.pyplot as plt
import cv2

# Ensure the image path is correct
image_path = 'images.jpg'

# Load the image using OpenCV
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    raise FileNotFoundError(f"The image at {image_path} could not be loaded.")

# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.title('Original Image')
plt.show()

# Reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1, 3))

# Convert to float type
pixel_vals = np.float32(pixel_vals)

# Define criteria for the k-means algorithm to stop running
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

# Perform k-means clustering with the number of clusters defined as 3
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert centers to 8-bit values
centers = np.uint8(centers)

# Map the centers to the segmented data
segmented_data = centers[labels.flatten()]

# Reshape the data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))

# Display the segmented image
plt.imshow(segmented_image)
plt.title('Segmented Image with k=3')
plt.show()
