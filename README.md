# Robot demonstrator

Software for the robot demonstrator at KU Leuven Bruges. This demonstrator showcases a ABB IRB1200 robot that picks circular objects using a RealSense D415 camera. For safety reasons a SICK Microscan 3 is added thats guards a safety zone around the robot setup. When launched, the robot sequentially picks all objects that have been placed in its picking zone and places them at a fixed spot outside the picking zone. Some basic image processing and kinematic calculations were used to obtain the result. 

## Folder structure

* config: contains the parameter files that are used in the scripts.
* data: contains the images that are used for calibration, as well as matrices resulted from calibration.
* rapid: contains the RAPID software that runs on the robot controller.
* scripts contains all scripts for executing the calibrations and running the main demo software.
* src: contains all source code including the image processing and driver functions for the camera and the robot.
* tests: contains all scripts for testing the functionalities in this package.

## Calibration 

For the robot calibration, the calibration script needs some images of the checkerboard mounted at the robot end-effector. Also the joint values of the robot at each image should be known. First, the intrinsics of the camera were computed using the 'abb_camera_intrinsic_calibration.py'-script. Second, the robot was moved to several random locations making sure that the checkerboard was clearly visible in the image taken by the camera. All jointt positions were saved using the 'abb_hand_eye_calibration_record_joints.py'-script. Third, the robot was moved through all joint positions and a corresponding image was taken of the checkerboard mounted at the robot end-effector. This was done using the 'abb_hand_eye_calibration_data_collection.py'-script. Next, the 'abb_hand_eye_calibration_data_preparation.py'-script was used to compute the transformation matrices from the camera frame to the target frame (T_ct) and from the robot baseframe to the end-effector frame (T_be). These matrices are saved for all images in a large matrix P and Q. Finally, the final transformation matrix from the robot baseframe to the camera frame (T_bc) was computed by solving the AX=XB system of equations using the 'abb_hand_eye_calibration.py'-script.

## Object detection

For object detection, a simple thresholding technique on HSV-values was used. The 'abb_hsv_calibration.py'-script was used to calibrate these threshold values based on images including the objects to detect. Further morphological transformations and contour detection with filters on the area of the contours were used. 

## Main

There are two main scripts. The 'main.py' script runs all code sequentially. The 'main_threaded.py'-script uses multithreading to separate the camera and robot functionality. This script also visualizes the detections of the camera and the picking area of the robot.

## A video of the setup was added to the repository
