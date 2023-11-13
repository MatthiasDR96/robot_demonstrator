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

	# Get mask
	mask = get_mask(image)
	
	print(mask)

	# Warp image
	#mask = cv2.warpPerspective(mask, M, (np.shape(image)[1], np.shape(image)[0]))

	# Get object pixel
	#center, radius = get_object_pixel(mask)