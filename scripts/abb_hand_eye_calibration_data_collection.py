# Imports
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from robot_demonstrator.Camera import Camera
from robot_demonstrator.ABB_IRB1200 import ABB_IRB1200

# Figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

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

    # Read robot
    cart = robot.con.get_cartesian() # [[X, Y, Z], [x, y, z, w]]
    T_be = t_from_xyz_r(cart[0][0], cart[0][1], cart[0][2], r_from_quat(cart[1][0], cart[1][1], cart[1][2], cart[1][3])) # Millimeters, radians
    robot_poses.append(T_be)

    # Read camera
    color, depth = cam.read()

    # Save image
    cv2.imwrite('./data/image' + str(i+1) + '.jpg', color) 

    # Plot image
    plt.imshow(color)
    plt.draw()
    plt.pause(0.1)

# Save data
np.save('./data/robot_poses.npy', np.array(robot_poses))




