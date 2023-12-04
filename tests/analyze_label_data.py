""" 
This script analyzes the computed XYZ-coordinates and compares them to the ground truth coordinates.

"""

# Imports 
import os
import numpy as np
import matplotlib.pyplot as plt
from robot_demonstrator.Camera import Camera
from robot_demonstrator.image_processing import *
from robot_demonstrator.plot import *
from robot_demonstrator.transformations import *
from robot_demonstrator.ABB_IRB1200 import ABB_IRB1200

# File name
file_name_labels = './data/labels/'
file_name_labels_gt = './data/labels_gt/'

# Create robot object (offline)
robot = ABB_IRB1200()

# Load T_bc (Transformation matrix from robot base frame to camera frame)
T_bc = np.load('./data/T_bc.npy')

# Get images list
labels = os.listdir(file_name_labels)

# Make data list
data = []
data_gt = []

# Loop over images
for index in range(len(labels)):

	# Open xyz_location to txt
	file = open(os.path.join(file_name_labels, labels[index]), 'r')
	line = file.readline().split(' ')
	if len(line) > 1:
		line = [float(x) for x in line]
		data.append(line)
	file.close()

	# Open ground truth xyz_location to txt
	file = open(os.path.join(file_name_labels_gt, labels[index]), 'r')
	line = file.readline().split(' ')
	if len(line) > 1:
		line = [float(x) for x in line]
		data_gt.append(line)
	file.close()

# Convert to numpy array
data = np.array(data)
data_gt = np.array(data_gt)

# Save data
np.save('./data/labels_data', data)
np.save('./data/labels_gt_data', data_gt)

# Plot data
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], 'g*') # Plot data
ax.scatter(data_gt[:,0], data_gt[:,1], data_gt[:,2], 'r*') # Plot ground truth
ax.scatter(0, 0, 0, 'k*') # Plot robot 
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_aspect('equal')
#plot_frame_t(T_bc, ax) # Plot camera frame
#robot.plot(ax, np.array([0, 0, 0, 0, 0, 0])) # Plot robot in home pose
plt.show()
