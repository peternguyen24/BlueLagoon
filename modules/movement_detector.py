import cv2
import numpy as np

class MovementDetector:
	def __init__(self):
		self.fps = None
		self.buffered_frames = None
		self.max_index = None
		self.current_index = None
		self.init = None

	def set_fps(self, fps):
		self.fps = fps
		self.max_index = int(fps // 2) # take 0.5 seconds of motion
		self.buffered_frames = np.zeros(self.max_index, dtype=(int,2))
		self.current_index = 0
		self.init = True

	# hand_position is (x,y) of palm centroid
	# buffered_frames [(x0,y0), (x1,y1), ..., (x29,y29)]
	def save_to_buffer(self, hand_position):
		if self.fps is None:
			return

		self.buffered_frames[self.current_index] = hand_position
		self.current_index = (self.current_index + 1) % self.max_index

	def match_motion(self):
		if self.fps is None:
			print('fps not set')
			return False
		return self.h_slide()

	# horizontal slide
	def h_slide(self):
		# print("current centroid coordinate")
		# print(current_x, current_y)
		# print("current index: ", self.current_index)
		# print("buffered centroid coordinate")
		# print(self.buffered_frames)

		detected = False
		current_x = self.buffered_frames[self.current_index][0]
		current_y = self.buffered_frames[self.current_index][1]

		for i in range(self.max_index):
			ind = (self.current_index + i) % self.max_index
			prev = (ind - 1 + self.max_index) % self.max_index
			x = self.buffered_frames[ind][0]
			y = self.buffered_frames[ind][1]
			x_prev = self.buffered_frames[prev][0]
			y_prev = self.buffered_frames[prev][1]
			
			if y < current_y-300 or y > current_y+300:
				return False
			if x < x_prev-10:
				return False
			else:
				detected = True

		return detected