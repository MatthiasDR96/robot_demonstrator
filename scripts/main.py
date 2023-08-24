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
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

# Define camera
cam = Camera()
cam.start()

# Define robot
robot = ABB_IRB1200("192.168.125.1")

# Reset robot
robot.con.set_speed([100, 50, 50, 50])
robot.con.set_joints([0, 0, 0, 0, 0, 0])
robot.con.set_dio(0)

# Load T_bc
T_bc = np.load('./data/T_bc.npy')

# Read frame
image, depth_image = cam.read()

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

	print(depth_pixel)

	# Transform 2D to 3D camera coordinates
	xcam, ycam, zcam = cam.intrinsic_trans(center, depth_pixel, cam.mtx)

	# Plot ball pixel
	cv2.circle(final_image, center, 5, (0, 0, 255), -1)
	cv2.circle(final_image, center, int(radius), (255, 0, 0), 5)
	center_as_string = ''.join(str(center))

	# Show
	plt.figure(1)
	plt.imshow(final_image)
	plt.show()

	# Transform camera coordinates to world coordinates
	p_bt = np.dot(T_bc, numpy.array([[xcam], [ycam], [zcam], [1]]))

	# Convert to transformation matrix
	T_bt = np.array([[-1, 0, 0, p_bt[0][0]],
					 [0, 1, 0, p_bt[1][0]],
					 [0, 0, -1, 60],
					 [0, 0, 0, 1]])

	print(p_bt)

	# Plot
	#plot_frame_t(T_bc, ax)
	#plot_frame_t(T_bt, ax)
	#plt.show()

	# Construct target pose
	error = [0, -15]
	offset = [-math.sqrt(200) + error[0], -math.sqrt(200) + error[1], 170]
	quat = list(quat_from_r(T_bt[:3,:3]))
	quat = [quat[3], quat[0], quat[1], quat[2]]
	xyz = [T_bt[0][3] + offset[0], T_bt[1][3] + offset[1], T_bt[2][3] + offset[2]]

	# Set pose 1
	robot.con.set_cartesian([xyz, quat])
	time.sleep(1)

	# Pick
	robot.con.set_dio(1)
	time.sleep(1)

	# Set pose 1 upper
	pose = pose = [[T_bt[0][3] + offset[0], T_bt[1][3] + offset[1], T_bt[2][3] + offset[2] + 40], quat]	
	robot.con.set_cartesian(pose)

	# Set pose 2 upper
	pose = [[288.46, -330.19, 240 + 40], quat]	
	robot.con.set_cartesian(pose)
	time.sleep(1)

	# Set pose 2
	pose = [[288.46, -330.19, 240], quat]	
	robot.con.set_cartesian(pose)
	time.sleep(1)

	# Place
	robot.con.set_dio(0)
	time.sleep(1)

	# Return to home
	robot.con.set_joints([0, 0, 0, 0, 0, 0])