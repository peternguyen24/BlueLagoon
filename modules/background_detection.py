import cv2
import numpy as np

class BackgroundDetection:
	def __init__(self):
		self.trained_background = False
		self.background = None

	def train_background(self, frame):
		if frame is not None:
			self.trained_background = True
			self.background = frame

			