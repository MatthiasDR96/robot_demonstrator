""" 
This script collects raw image data (RGB and Depth) from the camera and saves the images to the data/images folder.

"""

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from robot_demonstrator.Camera import Camera

# Define save folder
folder_rgb = "./data/images_rgb/"
folder_depth = "./data/images_depth/"

# Create camera object
cam = Camera()

# Start camera
cam.start()

# Loop
counter = 0
while(True): 
      
    # Read frame
    image, depth = cam.read()

    # If 'p' pressed, save data
    command = cv2.waitKey(1) & 0xFF
    if command == ord('p'):

        # Save RGB image to folder
        cv2.imwrite(folder_rgb + 'image_' + str(counter) + '.jpg', image)

        # Save depth image to folder
        np.save(folder_depth + 'image_' + str(counter), depth)

        # Print
        print("Saved image " + str(counter))

        # Increment counter
        counter += 1 

    # Show frame
    cv2.imshow('Frame', image)
    cv2.waitKey(2)

