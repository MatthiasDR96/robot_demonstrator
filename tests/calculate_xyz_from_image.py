""" 
This script reads the colleted raw image data (RGB and Depth) and computes the XYZ-coordinate of the detected object in robot base frame.

"""

# Imports 
import os
import cv2
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from robot_demonstrator.Camera import Camera
from robot_demonstrator.image_processing import *

# Suppress all warnings
warnings.filterwarnings('ignore')

# File name
file_name_images_rgb = './data/images_rgb/'
file_name_images_depth = './data/images_depth/'
file_name_labels = './data/labels/'

# Get images list
images = os.listdir(file_name_images_rgb)

# Create camera object
cam = Camera()

# Load T_bc (Transformation matrix from robot base frame to camera frame)
T_bc = np.load('./data/T_bc.npy')

# Load perspective matrix (calculated using the image_rectification_test.py file)
M = np.load('./data/perspective_transform.npy')

# Load error model
model = pickle.load(open('./data/error_model_20231215.sav', 'rb'))

# Loop over images
for index in range(len(images)):

	# Get image
	image = cv2.imread(os.path.join(file_name_images_rgb, images[index]))
	depth = np.load(os.path.join(file_name_images_depth, images[index].replace('.jpg', '.npy')))

	# Undistort image
	h, w = image.shape[:2]
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam.mtx, cam.dist, (w,h), 1, (w,h))
	mapx, mapy = cv2.initUndistortRectifyMap(cam.mtx, cam.dist, None, newcameramtx, (w, h), 5)
	image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

	# Warp image as if the camera took the image from above, perpendicular to the table
	warped_image = cv2.warpPerspective(image, M, (np.shape(image)[1], np.shape(image)[0]))

	# Get mask of the objects to detect (the threshold values can be calibrated using the 'abb_hsv_calibration.py'-file)
	mask = get_mask(warped_image)

	# Get object pixel from the mask
	data, contours_tmp = get_object_pixel(mask)

	# Get data
	center = data[0][0]
	radius = data[0][1]

	# If no object pixel found, continue
	if center is None: continue

	# Transform pixel on warped image back to original image
	new_pixel = np.dot(np.linalg.inv(M), np.array([[center[0]], [center[1]], [1]]))
	center = [int(new_pixel[0][0]/new_pixel[2][0]), int(new_pixel[1][0]/new_pixel[2][0])]

	# Plot ball pixel
	cv2.circle(image, center, 5, (0, 0, 255), -1)
	cv2.circle(image, center, int(radius), (255, 0, 0), 5)

	# Get pixel depth
	pixel_depth = depth[center[1], center[0]]

	# If there is depth data
	if pixel_depth is None: continue

	# Transform 2D to 3D camera coordinates
	xcam, ycam, zcam = cam.intrinsic_trans(center, pixel_depth, cam.mtx)

	# Transform camera coordinates to robot base frame using T_bc
	p_bt = np.dot(T_bc, np.array([[xcam], [ycam], [zcam], [1]])) # Position of the object in robot base frame

	# Create pick position for robot considering grip height and tool offset
	xyz = np.array([p_bt[0][0], p_bt[1][0], p_bt[2][0]]) 

	# Remove error using trained model
	#error = model.predict([xyz])
	#xyz = xyz - error
	#xyz = xyz[0]

	# Save xyz_location to txt
	file = open(file_name_labels + 'label_' + images[index].split('_')[-1].split('.')[0] + '.txt', 'w')
	file.write(str(xyz[0]) + " " + str(xyz[1]) + " " + str(xyz[2]))
	file.close()

	# Display the resulting frame 
	plt.imshow(image) 
	#plt.show()