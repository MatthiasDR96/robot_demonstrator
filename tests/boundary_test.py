# Import 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from robot_demonstrator.Camera import *

# Robot boundaries
xmin = 400
xmax = 630
ymin = -250
ymax = 250

# Load perspective matrix
M = np.load('./data/perspective_transform.npy')

# Load T_bc
T_bc = np.load('./data/T_bc.npy')

# Define camera
cam = Camera()
cam.start()

# Boundary coordinates
bound_coor = np.float32([[xmin, ymin, 0], [xmin, ymax, 0], [xmax, ymax, 0], [xmax, ymin, 0]]).reshape(-1, 3)

# Transform robot base frame coordinates to camera coordinates
T_cb = np.linalg.inv(T_bc)

# Transform 2D to 3D camera coordinates
imgpts_point, _ = cv2.projectPoints(bound_coor, T_cb[:3, :3], T_cb[0:3, 3], cam.mtx, cam.dist)
imgpts_point = imgpts_point.astype(int)

# Transform pixel back to original image plane
new_pixels = []
for point in imgpts_point:
    new_pixel = np.dot(M, np.append(point[0], 1))
    center = [int(new_pixel[0]/new_pixel[2]), int(new_pixel[1]/new_pixel[2])]
    new_pixels.append(center)
imgpts_point = new_pixels

# Read frame
image, depth_image = cam.read()

# Warp image
warped_image = cv2.warpPerspective(image, M, (np.shape(image)[1], np.shape(image)[0]))


cv2.line(warped_image, tuple(imgpts_point[0]), tuple(imgpts_point[1]), (0, 0, 255), 5)
cv2.line(warped_image, tuple(imgpts_point[1]), tuple(imgpts_point[2]), (0, 0, 255), 5)
cv2.line(warped_image, tuple(imgpts_point[2]), tuple(imgpts_point[3]), (0, 0, 255), 5)
cv2.line(warped_image, tuple(imgpts_point[3]), tuple(imgpts_point[0]), (0, 0, 255), 5)

plt.imshow(warped_image)
plt.show()