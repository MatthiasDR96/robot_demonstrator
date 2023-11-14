# Imports
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from robot_demonstrator.image_processing import *
from robot_demonstrator.Camera import *

# Define camera
cam = Camera()
cam.start()

# Read frame
image, depth_image = cam.read()

# Show image
plt.imshow(image)
plt.show()

# Define corners (manually defined)
corners1 = np.array([(100, 1080), (421, 0), (1378, 0), (1625, 1080)])
corners2 = np.array([(0, 1080), (0, 0), (1920, 0), (1920, 1080)])

# Get perspective transformation
M = cv2.getPerspectiveTransform(np.float32(np.resize(corners1, (4, 2))), np.float32(np.resize(corners2, (4, 2))))

# Save perspective transformatrion
np.save('./data/perspective_transform.npy', M)

# Warp image
dst = cv2.warpPerspective(image, M, (np.shape(image)[1], np.shape(image)[0]))

# Get mask
mask2 = get_mask(dst)

# Show mask
plt.imshow(mask2)
plt.show()

# Get center pixel
center, radius, _ = get_object_pixel(mask2)

# Plot ball pixel
cv2.circle(dst, center, 5, (0, 0, 255), -1)
cv2.circle(dst, center, int(radius), (255, 0, 0), 5)

# Show
plt.imshow(dst)
plt.show()

# Transform pixel back to original image plane
new_pixel = np.dot(np.linalg.inv(M), np.array([[center[0]], [center[1]], [1]]))
center = [int(new_pixel[0][0]/new_pixel[2][0]), int(new_pixel[1][0]/new_pixel[2][0])]

# Plot ball pixel
cv2.circle(image, center, 5, (0, 0, 255), -1)
cv2.circle(image, center, int(radius), (255, 0, 0), 5)
center_as_string = ''.join(str(center))

# Show
plt.imshow(image)
plt.show()
