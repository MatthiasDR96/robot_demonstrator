# Imports
import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from robot_demonstrator.plot import *
from robot_demonstrator.Camera import *
from robot_demonstrator.transformations import *
from robot_demonstrator.ABB_IRB1200 import ABB_IRB1200

# Figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define robot
robot = ABB_IRB1200()

# Define camera
cam = Camera()
cam.start()

# Load T_bc
T_bc = np.load('./data/T_bc.npy')

# Read frame
image, depth_image = cam.read()

# Get transformation matrix from camera to target
ret, corners2, rvecs, tvecs, T_ct = cam.extrinsic_calibration(image)

# Get HSV calibration params 
hsvfile = np.load('data/demo1_hsv.npy')

# Copy colour image
final_image = image.copy()

# Crop image
cropx = 200
cropy = 500
color_image = image[cropx:1000, cropy:1500, :]

# Gaussian blur
blurred_image = cv2.GaussianBlur(color_image, (7, 7), 0)

# Convert to hsv color space
hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

# Get mask
mask = cv2.inRange(hsv, np.array([hsvfile[0], hsvfile[2], hsvfile[4]]), np.array([hsvfile[1], hsvfile[3], hsvfile[5]]))

# Erode to close gaps
mask = cv2.erode(mask, None, iterations=2)

# Dilate to get original size
mask = cv2.dilate(mask, None, iterations=2)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# If ball is present
if len(contours) > 0:

	# Find contour with largest area
	maxcontour = max(contours, key=cv2.contourArea)

	# Find moments of the largest contour
	moments = cv2.moments(maxcontour)

	# Find ball center with moments
	center = [int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])]

	# Correct crop
	center[0] += cropy
	center[1] += cropx

	# Find radius of circle
	((x, y), radius) = cv2.minEnclosingCircle(maxcontour)

	# Get pixel depth
	depth_pixel = depth_image[center[1], center[0]]

	# Transform 2D to 3D camera coordinates
	xcam, ycam, zcam = cam.intrinsic_trans(center, depth_pixel, cam.mtx)

	# Plot ball pixel
	cv2.circle(final_image, center, 5, (0, 0, 255), -1)
	cv2.circle(final_image, center, int(radius), (255, 0, 0), 5)
	center_as_string = ''.join(str(center))

	# Show
	#plt.figure(1)
	#plt.imshow(final_image)
	#plt.show()

	# Transform camera coordinates to world coordinates
	p_tball = np.dot(np.linalg.inv(T_ct), numpy.array([[xcam], [ycam], [zcam], [1]]))

	# Convert to transformation matrix
	T_cball = np.array([[1, 0, 0, xcam],
						[0, 1, 0, ycam],
						[0, 0, 1, zcam],
						[0, 0, 0, 1]])

	print(p_tball)

	print(np.dot(T_bc, T_ct))

	# Plot
	robot.plot(ax, np.array([0, 0, 0, 0, 0, 0]))
	plot_frame_t(np.eye(4), ax)
	plot_frame_t(T_bc, ax)
	plot_frame_t(np.dot(T_bc, T_ct), ax)
	plot_frame_t(np.dot(T_bc, T_cball), ax)
	plt.show()

	