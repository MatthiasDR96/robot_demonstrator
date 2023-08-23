# Imports
import cv2
import numpy as np
from robot_demonstrator.Camera import Camera

# Get camera
cam = Camera()

# Loop over samples
while True:

	# Read camera
	color, depth = cam.read()

	# Get transformation matrix from camera to target
	ret, corners2, rvecs, tvecs, T_ct = cam.extrinsic_calibration(color)

	# Draw chessboard corners
	img = cv2.drawChessboardCorners(color, (cam.b, cam.h), corners2, ret)

	# Show image
	cv2.imshow('Frame', img)
	cv2.waitKey(0)

	# Save image
	cv2.imwrite('./data/camera_calibration_images/image' + str(i+1) + '.jpg', color)

	# 
	input("Press Enter to continue...")


