"""
This library imports the two functions for obtaining the center pixel of an object to detect.

"""

# Imports
import cv2
import numpy as np

# Get the mask of the object to detect
def get_mask(image):

    # Get HSV calibration params (calculated from the 'abb_hsv_calibration.py'-file)
    hsvfile = np.load('data/demo1_hsv_disk.npy')

    # Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (7, 7), 0)

    # Convert to hsv color space
    hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    # Get mask based on HSV-values
    mask = cv2.inRange(hsv, np.array([hsvfile[0], hsvfile[2], hsvfile[4]]), np.array([hsvfile[1], hsvfile[3], hsvfile[5]]))

    # Perform the opening operation
    size = 25
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*size+1, 2*size+1)))

    # Return mask
    return mask

# Get the object pixel of the object to detect
def get_object_pixel(mask):

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    contours_tmp = [contour for contour in contours if cv2.minEnclosingCircle(contour)[1] > 40 and cv2.minEnclosingCircle(contour)[1] < 75]
    
    # Filter contours by aspect ration
    contours_tmp = [contour for contour in contours_tmp if 1.0 < float(cv2.boundingRect(contour)[2])/cv2.boundingRect(contour)[3] < 2.0]
    
    # Calculate center and radius for all contours
    data = []
    for cnt in contours_tmp:
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        data.append(((int(x), int(y)), radius))

    # Return 
    return data, contours_tmp