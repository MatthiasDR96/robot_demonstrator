# Imports
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from robot_demonstrator.Camera import Camera

# Set save folder name
folder_name = './data/camera_calibration_images/'

# Create camera object
cam = Camera()

# Start camera
cam.start()

# Loop
count = 0
while count < 10:

    # Read images
    color_image, depth_image = cam.read()

    # To grayscale
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray_image, (cam.b, cam.h), None)

    # If corners found
    if ret == True:

        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

        # If key pressed
        command = cv2.waitKey(1) & 0xFF
        if command == ord('p'):

            # Save image
            cv2.imwrite(folder_name + 'image_' + str(count) + '.jpg', color_image)

            # Print 
            print("Saved image " + str(count))

            # Increment
            count += 1

        # Show chessbaord corners
        color_image = cv2.drawChessboardCorners(color_image, (cam.b, cam.h), corners2, ret)

    # Show frame
    cv2.imshow('Frame', color_image)
    cv2.waitKey(2)