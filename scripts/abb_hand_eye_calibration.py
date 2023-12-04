# Source: https://samarth-robo.github.io/blog/2020/11/18/robot_camera_calibration.html

# Imports
import numpy as np
import matplotlib.pyplot as plt
from robot_demonstrator.plot import *
from robot_demonstrator.calibrate import calibrate
from robot_demonstrator.ABB_IRB1200 import ABB_IRB1200

# Figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Setup robot
robot = ABB_IRB1200()

# Read matrices
Q = np.load('./data/Q.npy') # Transfromation matrices from robot base to robot end-effector in meters and radians
P = np.load('./data/P.npy') # Transfromation matrices from camera frame to target in meters and radians

# Create motion matrices
A, B = [], []
for i in range(len(Q)-1):
    A.append( np.dot(Q[i], np.linalg.inv(Q[i-1])) )
    B.append( np.dot(P[i], np.linalg.inv(P[i-1])) )

# Solve fro Tbc
Rx, tx = calibrate(A, B)

# Convert to transformation matrix from robot base to camera frame
T_bc = np.hstack((Rx, np.reshape(tx, (3,1))))
T_bc = np.vstack((T_bc, [0.0, 0.0, 0.0, 1.0]))

# Save
np.save('./data/T_bc.npy', T_bc)

# Compute error
error = np.dot(T_bc, A) - np.dot(T_bc, B)
print(sum(sum(sum(error))))

### To comput transformation matrix from end effector to target

# Create motion matrices
A, B = [], []
for i in range(len(Q)-1):
    A.append(np.dot(np.linalg.inv(Q[i]), Q[i-1])) 
    B.append(np.dot(np.linalg.inv(P[i]), P[i-1])) 

# Solve fro Tbc
Rx, tx = calibrate(A, B)

# Convert to transformation matrix from robot base to camera frame
T_et = np.hstack((Rx, np.reshape(tx, (3,1))))
T_et = np.vstack((T_et, [0.0, 0.0, 0.0, 1.0]))

# Plot frames
robot.plot(ax, np.array([0, 0, 0, 0, 0, 0]))
plot_frame_t(T_bc, ax)
#plt.show()