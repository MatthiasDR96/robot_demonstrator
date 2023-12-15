""" 
This script enables a visual test of the camera calibration. 
Put a calibration pattern in the camera field of view and run the script.
The script will show a visualisation of the robot frames, the camera frame, 
and the callibration frame. There can be visually checked if the relation between them reflects the reality.

"""

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from robot_demonstrator.plot import *
from robot_demonstrator.Camera import *
from robot_demonstrator.transformations import *
from robot_demonstrator.image_processing import *
from robot_demonstrator.ABB_IRB1200 import ABB_IRB1200

# Create camera object
cam = Camera()

# Start camera
cam.start()

# Create robot object (offline)
robot = ABB_IRB1200()

# Load T_bc (Transformation matrix from robot base frame to camera frame)
T_bc = np.load('./data/T_bc.npy')

# Read frame
image, depth_image = cam.read()

# Get mask of objects to detect
mask = get_mask(image)

# Get object pixel from the mask
data, contours_tmp = get_object_pixel(mask)

# Get data
center = data[0][0]
radius = data[0][1]

# Plot detected pixel
cv2.circle(image, center, 5, (0, 0, 255), -1)
cv2.circle(image, center, int(radius), (255, 0, 0), 5)
center_as_string = ''.join(str(center))

# Show image
plt.imshow(image)
plt.show()

# Get pixel depth
pixel_depth = depth_image[center[1], center[0]]
if pixel_depth is None or pixel_depth < 10: print("No depth information")

# Transform 2D to 3D camera coordinates
xcam, ycam, zcam = cam.intrinsic_trans(center, pixel_depth, cam.mtx)

# Extrinsic calibration
ret, corners2, rvecs, tvecs, T_ct = cam.extrinsic_calibration(image)

# Transform camera coordinates to world coordinates
p_to = np.dot(np.linalg.inv(T_ct), numpy.array([[xcam], [ycam], [zcam], [1]]))

# Convert to transformation matrix
T_co = np.array([[1, 0, 0, xcam],
				 [0, 1, 0, ycam],
				 [0, 0, 1, zcam],
				 [0, 0, 0, 1]])

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
robot.plot(ax, np.array([0, 0, 0, 0, 0, 0])) # Plot robot in home configuration
plot_frame_t(np.eye(4), ax) # Plot world frame
plot_frame_t(T_bc, ax) # Plot camera frame
plot_frame_t(np.dot(T_bc, T_ct), ax) # Plot callibration frame
plot_frame_t(np.dot(T_bc, T_co), ax) # Plot object frame
plt.show()

