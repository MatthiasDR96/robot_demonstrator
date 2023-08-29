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
robot.con.set_joints(joints)
#robot.con.set_cartesian([[400.0, 200.0, 600.1], [ 0.5, -0.5, 0.5, -0.5 ]]) # w, x, y, z --> send to [-0.5, 0.5, -0.5, 0.5]
#robot.con.set_cartesian([[400.0, 200.0, 600.1], [0, 0, 1, 0]]) # w, x, y, z --> send to [0, 1, 0, 0]
#robot.con.set_cartesian([[400.0, 200.0, 600.1], [0, 1, 0, 0]]) # w, x, y, z --> send to [1, 0, 0, 0]

time.sleep(4)

# Get joints
joints = np.radians(robot.con.get_joints())

# Get Cartesian pose
pose1_abb = robot.con.get_cartesian() # [[X, Y, Z], [w, x, y, z]]
T_be1 = t_from_xyz_r(pose1_abb[0][0], pose1_abb[0][1], pose1_abb[0][2], r_from_quat(pose1_abb[1][1], pose1_abb[1][2], pose1_abb[1][3], pose1_abb[1][0]))
print([pose1_abb[0], [pose1_abb[1][1], pose1_abb[1][2], pose1_abb[1][3], pose1_abb[1][0]]])

# Robot forward kinematics
T_be2 = robot.fkine(np.array(joints))
pose1_fkine = [[round(T_be2[0, 3], 1), round(T_be2[1, 3], 1), round(T_be2[2, 3], 1)], list(np.round(quat_from_r(T_be2[:3, :3]), 3))]
print(pose1_fkine)

# Plot robot
robot.plot(ax, np.array(joints))
plot_frame_t(T_be1, ax)
plt.show()




