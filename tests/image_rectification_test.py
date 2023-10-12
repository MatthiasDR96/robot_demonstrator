# Imports
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from robot_demonstrator.image_processing import *

# Read image
image1 =  cv2.imread('./data/perspective_1_Color.png')
image2 =  cv2.imread('./data/perspective_2_Color.png')
image3 =  cv2.imread('./data/perspective_1_ball_Color.png')

# Convert to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Get chessboard corners
ret, corners1 = cv2.findChessboardCornersSB(gray1, (9, 14), cv2.CALIB_CB_MARKER)
ret, corners2 = cv2.findChessboardCornersSB(gray2, (9, 14), cv2.CALIB_CB_MARKER)

# Show
#cv2.drawChessboardCorners(image1, (9,14), corners1,ret)
#plt.imshow(image1)
#plt.show()

#cv2.drawChessboardCorners(image2, (9,14), corners2,ret)
#plt.imshow(image2)
#plt.show()

corners1 = np.array([corners1[0], corners1[8], corners1[-1], corners1[-9]])
corners2 = np.array([corners2[0], corners2[8], corners2[-1], corners2[-9]])

# Get perspective transformation
M = cv2.getPerspectiveTransform(np.float32(np.resize(corners1, (4, 2))), np.float32(np.resize(corners2, (4, 2))))
np.save('./data/perspective_transform.npy', M)

# Get mask
mask = get_mask(image3)
mask2 = get_mask(image1)

# Warp image
dst = cv2.warpPerspective(mask, M, (np.shape(image1)[1], np.shape(image1)[0]))

# Get center pixel
center, radius = get_object_pixel(dst)
center2, radius = get_object_pixel(mask2)

# Plot ball pixel
cv2.circle(dst, center, 5, (0, 0, 255), -1)
cv2.circle(dst, center, int(radius), (255, 0, 0), 5)
center_as_string = ''.join(str(center))

# Show
plt.imshow(dst)
plt.show()

# Transform pixel back to original image plane
new_pixel = np.dot(np.linalg.inv(M), np.array([[center[0]], [center[1]], [1]]))
center = [int(new_pixel[0][0]/new_pixel[2][0]), int(new_pixel[1][0]/new_pixel[2][0])]

print (center)
print(center2)

# Plot ball pixel
cv2.circle(image3, center, 5, (0, 0, 255), -1)
cv2.circle(image3, center, int(radius), (255, 0, 0), 5)
center_as_string = ''.join(str(center))

# Show
plt.imshow(image3)
plt.show()
