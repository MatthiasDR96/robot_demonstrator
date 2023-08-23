# Imports
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from robot_demonstrator.Motion import *
from robot_demonstrator.plot import *
from robot_demonstrator.transformations import *
from robot_demonstrator.ABB_IRB1200 import ABB_IRB1200

# Figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define robot
robot = ABB_IRB1200("192.168.125.1")

# Set speed
#robot.con.set_speed(speed=[10, 10, 10, 10])

# Get info
print(robot.con.get_tool())
print(robot.con.get_robotinfo())

# Joint configuration
joints = [0, 0, 0, 0, 0, 0]

# Set joint goal
print(robot.con.set_joints(joints))

# Get joints
joints = np.radians(robot.con.get_joints())
print(joints)

# Get Cartesian pose
cart = robot.con.get_cartesian() # [[X, Y, Z], [x, y, z, w]]
print(cart)

# Convert to transformation matrix
T_be1 = t_from_xyz_r(cart[0][0], cart[0][1], cart[0][2], r_from_quat(cart[1][0], cart[1][1], cart[1][2], cart[1][3]))
print(T_be1)

# Robot forward kinematics
T_be2 = robot.fkine(np.array(joints))
print(T_be2)

# Plot robot
robot.plot(ax, np.array(joints))
plot_frame_t(T_be1, ax)
plt.show()




