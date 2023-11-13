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

# Load T_bc
M = np.load('./data/perspective_transform.npy')

# Loop
while True:

	# Read frame
	image, depth_image = cam.read()

	# Warp image
	#image = cv2.warpPerspective(image, M, (np.shape(image)[1], np.shape(image)[0]))

	# Get mask
	mask = get_mask(image)

	# Get object pixel
	center, radius, contours_tmp = get_object_pixel(mask)

	cv2.drawContours(image, contours_tmp, -1, (0,255,0), 3)

	plt.imshow(image)
	plt.draw()
	plt.pause(0.1)