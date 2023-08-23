# Imports
import numpy as np
from robot_demonstrator.ABB_IRB1200 import ABB_IRB1200

# Init
joint_values = np.load('./data/joint_values.npy')

# Define robot
robot = ABB_IRB1200("192.168.125.1")

# Get positions
joints = np.radians(np.array(robot.con.get_joints())) # Radians
joint_values = np.vstack((joint_values, joints))

# Print
print(joint_values)

# Set joint goal 3
np.save('./data/joint_values.npy', joint_values)
