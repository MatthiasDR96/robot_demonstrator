# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from robot_demonstrator.plot import *
from robot_demonstrator.Camera import Camera
from robot_demonstrator.ABB_IRB1200 import ABB_IRB1200

# Draws target axis
def draw_axis(img, imgpts):
    img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    text_pos = (imgpts[0].ravel() + np.array([3.5, -7])).astype(int)
    cv2.putText(img, 'X', tuple(text_pos), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    text_pos = (imgpts[1].ravel() + np.array([3.5, -7])).astype(int)
    cv2.putText(img, 'Y', tuple(text_pos), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    text_pos = (imgpts[2].ravel() + np.array([3.5, -7])).astype(int)
    cv2.putText(img, 'Z', tuple(text_pos), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    text_pos = (imgpts[3].ravel() + np.array([3.5, -7])).astype(int)
    return img

# Get camera
cam = Camera()

# Setup robot
robot = ABB_IRB1200()

# Get joint values
joint_values_list = np.load('./data/joint_values.npy')

# Loop
Q = [] # Transfromation matrices from robot base to robot end-effector in millimeters and radians
P = [] # Transfromation matrices from camera frame to target in millimeters and radians
for i in range(len(joint_values_list)):

    # Joint_values
    joint_values = joint_values_list[i]

    # Read image
    img = cv2.imread('./data/image' + str(i+1) + '.jpg')

    # Get transformation matrix from camera to target
    ret, corners2, rvecs, tvecs, T_ct = cam.extrinsic_calibration(img)

    # Get transformation matrix from robot base to end effector
    T_be = robot.fkine(joint_values) # Millimeters, radians

    # Draw chessboard
    img = cv2.drawChessboardCorners(img, (cam.b, cam.h), corners2, ret)

    # Project 3D axis points to pixel coordinates
    axis = np.float32([[60, 0, 0], [0, 60, 0], [0, 0, 60], [0, 0, 0]])
    imgpts_axis, _ = cv2.projectPoints(axis, T_ct[:3, :3], T_ct[0:3, 3], cam.mtx, cam.dist)
    image = draw_axis(img, imgpts_axis.astype(int))

    # Append matrices
    P.append(T_ct)
    Q.append(T_be)

    # Get T_et
    T_et = np.array([[0, 1, 0, 0.080], 
                    [-1, 0, 0, 0.077], 
                    [0, 0, 1, 0], 
                    [0, 0, 0, 1]])

    # Get T_bc
    T_bc = np.dot(np.dot(T_be, T_et), np.linalg.inv(T_ct))

    # Plot figure
    plt.figure(2)
    plt.imshow(img)

    # Plot robot
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    robot.plot(ax, np.array(joint_values))
    plot_frame_t(np.dot(T_be, T_et), ax)
    plot_frame_t(T_bc, ax)
    plt.show()

# Save matrices
np.save('./data/Q.npy', Q)
np.save('./data/P.npy', P)







