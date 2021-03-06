import cv2
import numpy as np
from . import image_analysis

class FrameProcessor:
	def __init__(self):
		self.row_ratio = None
		self.col_ratio = None

	def resize(self, frame):
		rows,cols,_ = frame.shape

		ratio = float(cols)/float(rows)
		new_rows = 400
		new_cols = int(ratio*new_rows)

		self.row_ratio = float(rows)/float(new_rows)
		self.col_ratio = float(cols)/float(new_cols)
		
		resized = cv2.resize(frame, (new_cols, new_rows))	
		return resized

	def flip(self, frame):
		flipped = cv2.flip(frame, 1)
		return flipped	

	def draw_final(self, frame, HD, BD):
		origin_frame = frame
		# frame = self.remove_static_bg(frame, BD)
		hand_masked, bw= image_analysis.apply_hist_mask(frame, HD.hand_hist)

		hand_contour, bw_hand_frame = self.extract_hand_contour(hand_masked)

		if hand_contour is not None:
			palm_radius, palm_center = HD.find_palm(hand_contour)
			hull = image_analysis.hull(hand_contour)
			fingers = HD.find_fingers(hull, palm_center, palm_radius)
			centroid = image_analysis.centroid(hand_contour) # currently not use
			defects = image_analysis.defects(hand_contour)

			if centroid is not None and defects is not None and len(defects) > 0:   
				farthest_point = image_analysis.farthest_point(defects, hand_contour, centroid)

				if farthest_point is not None:
					self.plot_farthest_point(origin_frame, farthest_point)
					self.plot_hull(origin_frame, hull)
					self.plot_palm_circle(origin_frame, palm_center, palm_radius)
					self.plot_fingers(origin_frame, fingers)

		frame_final = np.vstack([origin_frame, bw])
		return frame_final

	def extract_hand_contour(self, frame):
		max_contour = None
		bw_frame = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
		
		contours = image_analysis.contours(frame)
		if contours is not None and len(contours) > 0:
			max_contour = image_analysis.max_contour(contours)
		cv2.drawContours(bw_frame, [max_contour], -1, (0,255,0), 3)
		
		# strengthen and lighten the contour
		kernel = np.ones((5,5),np.uint8)
		bw_frame = cv2.dilate(bw_frame,kernel,iterations = 1)
		bw_frame = cv2.erode(bw_frame,kernel,iterations = 1)

		contours = image_analysis.contours(bw_frame)
		if contours is not None and len(contours) > 0:
			max_contour = image_analysis.max_contour(contours)

		return max_contour, bw_frame





	def remove_static_bg(self, frame, BD):
		fg_mask = cv2.absdiff(frame, BD.static_background)
		fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2GRAY)
		ret, fg_mask = cv2.threshold(fg_mask,10,255,cv2.THRESH_BINARY)
		frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
		return frame

	def get_dynamic_fg(self, frame, BD):
		fg_mask = BD.fgbg.apply(frame)
		frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
		return frame
 
	def plot_defects(self, frame, defects, contour):
		if len(defects) > 0:
			for i in xrange(defects.shape[0]):
				s,e,f,d = defects[i,0]
				start = tuple(contour[s][0])
				end = tuple(contour[e][0])
				far = tuple(contour[f][0])               
				cv2.circle(frame, start, 5, [255,0,255], -1)

	def plot_farthest_point(self, frame, point):
		cv2.circle(frame, point, 5, [0,0,255], -1)			

	
	def plot_centroid(self, frame, point):
		cv2.circle(frame, point, 5, [255,0,0], -1)

	
	def plot_hull(self, frame, hull):
		cv2.drawContours(frame, [hull], 0, (255,0,0), 2)	

	def plot_fingers(self, frame, fingers):
		for i in range(len(fingers)):
			cv2.circle(frame, (fingers[i][0],fingers[i][1]), 5, [255,0,0], -1)

	def plot_contours(self, frame, contours):
		cv2.drawContours(frame, contours, -1, (0,255,0), 3)				
	def plot_palm_circle(self, frame, center, radius):
		cv2.circle(frame, center, radius, [255,0,0], 2)
		cv2.circle(frame, center, 5, [255,0,0], -1)