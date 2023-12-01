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

    # Erode to close gaps
    mask = cv2.erode(mask, None, iterations=2)

    # Dilate to get original size
    mask = cv2.dilate(mask, None, iterations=2)
    
    # Return mask
    return mask

# Get the object pixel of the object to detect
def get_object_pixel(mask):

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If there is a contour
    if len(contours) > 0:

        # Filter contours by area
        contours_tmp = [contour for contour in contours if cv2.contourArea(contour) > 4500]
        contours_tmp = [contour for contour in contours_tmp if cv2.contourArea(contour) < 9000]

        # Check if contours are left
        if len(contours_tmp) > 0:

            # Sort contours on x-position
            contours_tmp.sort(key=lambda ctr: cv2.minEnclosingCircle(ctr[0])[0][0])

            # Select most left one
            selected_contour = contours_tmp[0]

            # Find center point and radius of contour
            ((x, y), radius) = cv2.minEnclosingCircle(selected_contour)

            # Return 
            return (int(x), int(y)), radius, contours_tmp
        
        # No contour found
        else:
            print("No contour found! (Contours in mask not in area range)")
            return None, None, None

    # No contour found
    else:
        print("No contour found! (No contour in mask)")
        return None, None, None
