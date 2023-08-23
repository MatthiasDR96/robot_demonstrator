# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from robot_demonstrator.Camera import Camera

# Get camera
cam = Camera()
cam.start()

# Declare arrays to store object points and image points from all the images.
imgpoints = [] 
objpoints = []

# Get object points for 1 chessboard
objp = np.zeros((cam.h * cam.b, 3), np.float32)
objp[:, :2] = np.mgrid[0:cam.b, 0:cam.h].T.reshape(-1, 2)
objp = cam.size * objp

# Loop
count = 0
while count < 10:

    print(count)

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

        # Append
        objpoints.append(objp)
        imgpoints.append(corners2)

        # Show chessbaord corners
        color_image = cv2.drawChessboardCorners(color_image, (cam.b, cam.h), corners2, ret)

    # Show image
    plt.imshow(color_image)
    plt.draw()
    plt.pause(0.1)

# Calculate intrinsics
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1], None, None)
print("intrinsieke matrix:\n")
print(mtx)
print("distortie:\n")
print(dist)

# Calculate reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print("\ntotal error:\n", mean_error / len(objpoints))