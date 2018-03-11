import cv2
import numpy as np

class BackgroundDetection:
	def __init__(self):
		self.trained_background = False
		self.static_background = None
		self.fgbg = cv2.createBackgroundSubtractorMOG2(500)

	def set_static_background(self, frame):
		if frame is not None:
			self.trained_background = True
			self.static_background = frame




			