# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_mask(image):

    # Get HSV calibration params 
    hsvfile = np.load('data/demo1_hsv_disk.npy')

    # Crop properties
    xstart, ystart, xend, yend = 500, 200, 1500, 1000

    # Crop image
    color_image = image[200:1000, 500:1500, :]

    # Gaussian blur
    blurred_image = cv2.GaussianBlur(color_image, (7, 7), 0)

    # Convert to hsv color space
    hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    # Get mask
    mask = cv2.inRange(hsv, np.array([hsvfile[0], hsvfile[2], hsvfile[4]]), np.array([hsvfile[1], hsvfile[3], hsvfile[5]]))

    # Erode to close gaps
    mask = cv2.erode(mask, None, iterations=2)

    # Dilate to get original size
    mask = cv2.dilate(mask, None, iterations=2)

    # Get original dimensions of image
    new_mask = np.zeros(np.shape(image)[0:2]).astype(np.uint8)
    new_mask[ystart:yend, xstart:xend] = mask

    # Return mask
    return new_mask


def get_object_pixel(mask):

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If there is a contour
    if len(contours) > 0:

        # Find contour with largest area
        maxcontour = max(contours, key=cv2.contourArea)

        # Find radius of circle
        ((x, y), radius) = cv2.minEnclosingCircle(maxcontour)

        # Return 
        return (int(x), int(y)), radius

    # No contour found
    else:
        print("No contour found")
        return None
