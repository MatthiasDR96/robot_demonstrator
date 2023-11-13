# Imports
import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from robot_demonstrator.plot import *
from robot_demonstrator.Camera import *
from robot_demonstrator.transformations import *
from robot_demonstrator.image_processing import *
from robot_demonstrator.ABB_IRB1200 import ABB_IRB1200

# Define camera
cam = Camera()
cam.start()

# Define robot
robot = ABB_IRB1200("192.168.125.1")
robot.start()

# Load T_bc
T_bc = np.load('./data/T_bc.npy')

# Load perspective matrix
M = np.load('./data/perspective_transform.npy')

# Define rotations and positions
grip_heigt_ball = 60
grip_heigth_disk = 8
quat = list(quat_from_r(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])))
quat = [quat[3], quat[0], quat[1], quat[2]]
grip_height = grip_heigth_disk
pose2 = [483.0, -290.9, 240] # Place pose
offset1 = np.array([0, 0, 40]) # Pick and place offset
offset2 = np.array([-math.sqrt(200), -math.sqrt(200), 169]) # Tool offset
error = [15, 10, 0] # Error in system --> to be reduced

# Loop
while True:

	# Read frame
	image, depth_image = cam.read()

	# Get mask
	mask = get_mask(image)

	# Warp image
	mask = cv2.warpPerspective(mask, M, (np.shape(image)[1], np.shape(image)[0]))

	# Get object pixel
	center, radius = get_object_pixel(mask)

	# If no mask found, continue
	if center is None: continue

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

	# Get pixel depth
	pixel_depth = depth_image[center[1], center[0]]
	if pixel_depth is None: print("No depth information")

	# The xy pixel of the detected circle center equals this of the center of the ball as
	# the ray goes through this ball, thus 'center' is correct. The pixel depth however should 
	# be increased with the ball radius to obtain the 3D coordinate of the centre. 
	#pixel_depth += 30

	# Transform 2D to 3D camera coordinates
	xcam, ycam, zcam = cam.intrinsic_trans(center, pixel_depth, cam.mtx)

	# Transform camera coordinates to robot base frame
	p_bt = np.dot(T_bc, numpy.array([[xcam], [ycam], [zcam], [1]]))
	xyz_base = np.array([p_bt[0][0], p_bt[1][0], grip_height]) + offset2 + error

	# Set pose 1 upper
	robot.con.set_cartesian([xyz_base + offset1, quat])
	time.sleep(1)

	# Set pose 1
	robot.con.set_cartesian([xyz_base, quat])
	time.sleep(1)

	# Pick
	robot.con.set_dio(1)
	time.sleep(1)

	# Set pose 1 upper
	robot.con.set_cartesian([xyz_base + offset1, quat])
	time.sleep(1)

	# Set pose 2 upper
	robot.con.set_cartesian([pose2 + offset1, quat])
	time.sleep(1)

	# Set pose 2
	robot.con.set_cartesian([pose2, quat])
	time.sleep(1)

	# Place
	robot.con.set_dio(0)
	time.sleep(1)

	# Return to home
	robot.con.set_joints([0, 0, 0, 0, 0, 0])
	time.sleep(1)