# Imports
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from robot_demonstrator.image_processing import *

# Read image
image = np.load('./data/color.npy')

# Get perspective transformed image
rect_image, M = get_perspective_image(image)

# Get mask
mask = get_mask(rect_image)

# Get center pixel
center, radius = get_object_pixel(mask)

# Transform pixel back to original image plane
new_pixel = np.dot(np.linalg.inv(M), np.array([[center[0]], [center[1]], [1]]))
center = [int(new_pixel[0][0]), int(new_pixel[1][0])]

# Plot ball pixel
cv2.circle(image, center, 5, (0, 0, 255), -1)
cv2.circle(image, center, int(radius), (255, 0, 0), 5)
center_as_string = ''.join(str(center))

# Show
plt.imshow(image)
plt.show()
