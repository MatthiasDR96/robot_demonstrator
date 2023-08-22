# Imports
import os
import cv2
import numpy
import pyrealsense2 as realsense

class Camera:

	def __init__(self):

		# Camera configuration properties
		self.color_resolution = (1920, 1080)
		self.depth_resolution = (1280, 720)
		self.frames_per_second = 30

		# Camera connection properties
		self.conn = None
		self.conf = None
		self.align = None

		# Camera calibration properties
		self.mtx = numpy.load('./data/intrinsics.npy')
		self.dist = numpy.load('./data/distortion.npy')

		# Chessboard properties
		self.h = 14
		self.b = 9
		self.size = 17.4 # mm

	def start(self):

		# Connect
		self.conn = realsense.pipeline()

		# Config
		self.conf = realsense.config()
		self.conf.enable_device('821312060307')
		self.conf.enable_stream(realsense.stream.depth, self.depth_resolution[0], self.depth_resolution[1], realsense.format.z16, self.frames_per_second)
		self.conf.enable_stream(realsense.stream.color, self.color_resolution[0], self.color_resolution[1], realsense.format.bgr8, self.frames_per_second)
		
		# Start streaming
		self.conn.start(self.conf)

		# Align images
		self.align = realsense.align(realsense.stream.color)

	def read(self):

		# Wait for image
		frames = self.conn.wait_for_frames()

		# Align images
		aligned_frames = self.align.process(frames)

		# Retreive images
		color_frame = aligned_frames.get_color_frame()
		depth_frame = aligned_frames.get_depth_frame()

		# Convert to arrays
		depth = numpy.asanyarray(depth_frame.get_data())
		color = numpy.asanyarray(color_frame.get_data())

		return color, depth

	# Get depth of pixel
	def get_pixel_depth(self, image, pixel):
		depth = image[pixel[1], pixel[0]]
		return depth

	# Extrinsic calibration
	def extrinsic_calibration(self, img):

		# termination criteria
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

		# objectpunten van het schaakbord voorbereiden
		objp = numpy.zeros((self.b * self.h, 3), numpy.float32)
		objp[:, :2] = numpy.mgrid[0:self.b, 0:self.h].T.reshape(-1, 2)
		objp = self.size * objp

		# Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Get chessboard corners
		ret, corners = cv2.findChessboardCornersSB(gray, (self.b, self.h), cv2.CALIB_CB_MARKER)
	
		# If corners are found
		if ret == True:
			
			# Refine corners
			corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

			# Extrinsic calibration
			ret, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners2, self.mtx, self.dist)

			# Get extrinsic matrix
			rvecs_matrix = cv2.Rodrigues(rvecs)[0]
			extrinsics = numpy.hstack((rvecs_matrix, tvecs))
			extrinsics = numpy.vstack((extrinsics, [0.0, 0.0, 0.0, 1.0]))

			return ret, corners2, rvecs, tvecs, extrinsics

		# If corners not found
		else:
			return None, None, None, None, None


	# Covert 2D to 3D cooridnates
	def intrinsic_trans(self, pixel, z, mtx):
		if (z):
			x = (pixel[0] - mtx[0, 2]) / mtx[0, 0] * z
			y = (pixel[1] - mtx[1, 2]) / mtx[1, 1] * z
			return x, y, z
		else:
			return None, None, None

	# Convert camera to world coordinates
	def extrinsic_trans(self, depth, x, y, z, ext):
		if (depth):
			mat = numpy.array([[x], [y], [z], [1]])
			inv = ext #numpy.linalg.inv(ext)
			world = numpy.dot(inv, mat)
			xw, yw, zw = world[0, 0], world[1, 0], world[2, 0],
			newx = yw
			newy = xw
			newz = -zw
			return newx, newy, newz
		else:
			return None, None, None

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