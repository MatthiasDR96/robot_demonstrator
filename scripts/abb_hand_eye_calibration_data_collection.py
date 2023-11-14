# Imports
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from robot_demonstrator.Camera import Camera
from robot_demonstrator.transformations import *
from robot_demonstrator.image_processing import *
from robot_demonstrator.ABB_IRB1200 import ABB_IRB1200

# Get joint values
joint_values = np.load('./data/joint_values.npy')

# Setup camera
cam = Camera()
cam.start()

# Setup robot
robot = ABB_IRB1200("192.168.125.1")

# Loop over samples
robot_poses = []
for i in range(len(joint_values)):

    # Move robot
    robot.con.set_joints(joint_values[i])

    # Sleep
    time.sleep(1)

    # Read camera
    color, depth = cam.read()

    # Save image
    cv2.imwrite('./data/camera_robot_calibration_images/image' + str(i+1) + '.jpg', color) 

    # Plot image
    plt.imshow(color)
    plt.draw()
    plt.pause(1)




