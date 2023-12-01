# Imports
import cv2
import time
import math
import time
import _thread
import numpy as np
import matplotlib.pyplot as plt
from robot_demonstrator.plot import *
from robot_demonstrator.Camera import *
from robot_demonstrator.transformations import *
from robot_demonstrator.image_processing import *
from robot_demonstrator.ABB_IRB1200 import ABB_IRB1200

# Create camera object
cam = Camera()

# Start camera
cam.start()

# Create camera object (online)
robot = ABB_IRB1200("192.168.125.1")

# Start robot
robot.start()

# Load T_bc (Transformation matrix from robot base frame to camera frame)
T_bc = np.load('./data/T_bc.npy')

# Load perspective matrix (calculated using the image_rectification_test.py file)
M = np.load('./data/perspective_transform.npy')

# Define pick and place orientation
quat = list(quat_from_r(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]))) # Quaternion of the pick and place orientation
quat = [quat[3], quat[0], quat[1], quat[2]] # Convert [x y z w] to [w x y z]

# Define fixed z-position to pick
grip_height = 7

# Define place position
pose_place = [483.0, 290.9, 240] 

# Define offsets
offset1 = np.array([0, 0, 40]) # Offset above the pick and place poses
tool_offset = np.array([-math.sqrt(200), -math.sqrt(200), 170]) # Tool offset (Translation from robot end effector to TCP)

# Define error
error = [15, 10, 0] # Error in system (visually determined) --> to be reduced

# Robot boundaries
xmin = 400 # Minimal Cartesian x-position
xmax = 630 # Maximal Cartesian x-position
ymin = -250 # Minimal Cartesian y-position
ymax = 250 # Maximal Cartesian y-position

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Global params
global xyz_base
xyz_base = None

# Robot task thread function
def robot_task(name):

	# Define global variable
	global xyz_base

	# Loop
	while True:

		# Check if there is a valid position
		if xyz_base is not None:

			# Fixed last position
			xyz_base_tmp = xyz_base

			# Final safety layer on Cartesian position
			if not (xyz_base_tmp[0] > xmax or xyz_base_tmp[0] < xmin or xyz_base_tmp[1] > ymax or xyz_base_tmp[1] < ymin):

				# Set pick pose upper
				robot.con.set_cartesian([xyz_base_tmp + offset1, quat])
				time.sleep(1)

				# Set pick pose
				robot.con.set_cartesian([xyz_base_tmp, quat])
				time.sleep(1)

				# Set pick DIO
				robot.con.set_dio(1)
				time.sleep(1)

				# Set pick pose upper
				robot.con.set_cartesian([xyz_base_tmp + offset1, quat])
				time.sleep(1)

				# Set place pose upper
				robot.con.set_cartesian([pose_place + offset1, quat])
				time.sleep(1)

				# Set place pose
				robot.con.set_cartesian([pose_place, quat])
				time.sleep(1)

				# Set place DIO
				robot.con.set_dio(0)
				time.sleep(1)

				# Set home position
				robot.con.set_joints([0, 0, 0, 0, 0, 0])
				time.sleep(1)
	
# Define a function for the thread
def camera_task(name):

	# Global
	global xyz_base

	# Loop
	while True:

		# Read frame
		image, depth_image = cam.read()

		# Warp image as if the camera took the image from above, perpendicular to the table
		warped_image = cv2.warpPerspective(image, M, (np.shape(image)[1], np.shape(image)[0]))

		# Get mask of the objects to detect (the threshold values can be calibrated using the 'abb_hsv_calibration.py'-file)
		mask = get_mask(warped_image)

		# Get object pixel from the mask
		center, radius, contours_tmp = get_object_pixel(mask)

		# If no object pixel found, continue
		if center:

			# Transform pixel on warped image back to original image
			new_pixel = np.dot(np.linalg.inv(M), np.array([[center[0]], [center[1]], [1]]))
			center = [int(new_pixel[0][0]/new_pixel[2][0]), int(new_pixel[1][0]/new_pixel[2][0])]

			# Plot ball pixel
			cv2.circle(warped_image, center, 5, (0, 0, 255), -1)
			cv2.circle(warped_image, center, int(radius), (255, 0, 0), 5)

			# Get pixel depth
			pixel_depth = depth_image[center[1], center[0]]

			# If there is depth data
			if pixel_depth:

				# Transform 2D to 3D camera coordinates
				xcam, ycam, zcam = cam.intrinsic_trans(center, pixel_depth, cam.mtx)

				# Transform camera coordinates to robot base frame using T_bc
				p_bt = np.dot(T_bc, numpy.array([[xcam], [ycam], [zcam], [1]])) # Position of the object in robot base frame

				# Create pick position for robot considering grip height and tool offset
				xyz_base = np.array([p_bt[0][0], p_bt[1][0], grip_height]) + tool_offset + error

			# Wrong picking pojnt
			else: xyz_base = None
		
		# Wrong picking point
		else: xyz_base = None


		### The code below is to calculate the pixel coordinates of the robot bounds

		# Boundary coordinates in robot base frame
		bound_coor = np.float32([[xmin, ymin, 0], [xmin, ymax, 0], [xmax, ymax, 0], [xmax, ymin, 0]]).reshape(-1, 3)

		# Transform robot base frame coordinates to camera coordinates
		T_cb = np.linalg.inv(T_bc)

		# Transform 3D to 2D pixel coordinates
		imgpts_point, _ = cv2.projectPoints(bound_coor, T_cb[:3, :3], T_cb[0:3, 3], cam.mtx, cam.dist)
		imgpts_point = imgpts_point.astype(int)

		# Transform pixel back to original image plane
		new_pixels = []
		for point in imgpts_point:
			new_pixel = np.dot(M, np.append(point[0], 1))
			center = [int(new_pixel[0]/new_pixel[2]), int(new_pixel[1]/new_pixel[2])]
			new_pixels.append(center)
		imgpts_point = new_pixels

		# Draw robot boundary
		cv2.line(warped_image, tuple(imgpts_point[0]), tuple(imgpts_point[1]), (0, 0, 255), 5)
		cv2.line(warped_image, tuple(imgpts_point[1]), tuple(imgpts_point[2]), (0, 0, 255), 5)
		cv2.line(warped_image, tuple(imgpts_point[2]), tuple(imgpts_point[3]), (0, 0, 255), 5)
		cv2.line(warped_image, tuple(imgpts_point[3]), tuple(imgpts_point[0]), (0, 0, 255), 5)
		

		### The code below is to show the results on the screen

		# Display the raw image with bounding box
		final_image = cv2.resize(warped_image, (int(1920/2), int(1080/2)))  
		cv2.imshow('frame1', final_image)
		cv2.resizeWindow("frame1", (int(1920/2), int(1080/2)))  
		cv2.moveWindow("frame1", 0, 0)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break 

		# Display the warped image with contours
		final_image = cv2.resize(cv2.drawContours(warped_image.copy(), contours_tmp, -1, (0,255,0), 3), (int(1920/2), int(1080/2)))  
		cv2.imshow('frame2', final_image)
		cv2.resizeWindow("frame2", (int(1920/2), int(1080/2)))  
		cv2.moveWindow("frame2", 0, int(1080/2))
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break 

		# Display the mask
		final_image = cv2.resize(mask, (int(1920/2), int(1080/2)))  
		cv2.imshow('frame3', final_image)
		cv2.resizeWindow("frame3", (int(1920/2), int(1080/2)))  
		cv2.moveWindow("frame3", int(1920/2), 0)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break 

		# Display the ldepth color map
		final_image = cv2.resize(cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET), (int(1920/2), int(1080/2)))  
		cv2.imshow('frame4', final_image)
		cv2.resizeWindow("frame4", (int(1920/2), int(1080/2)))  
		cv2.moveWindow("frame4", int(1920/2), int(1080/2))
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break 


if __name__ == "__main__":

	# Create two threads
	try:
		_thread.start_new_thread(camera_task, ("Thread-1", ) )
		_thread.start_new_thread(robot_task, ("Thread-2", ) )
	except:
		print ("Error: unable to start thread")

	# Loop threads
	while True:
		pass