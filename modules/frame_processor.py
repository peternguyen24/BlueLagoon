import cv2
import numpy as np
from . import image_analysis
import win32api

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

	def draw_final(self, frame, HD, BD, MD, frame_number):
		origin_frame = frame
		frame = self.remove_static_bg(frame, BD)
		hand_masked, bw_hand_masked = image_analysis.apply_hist_mask(frame, HD.hand_hist)

		contours = image_analysis.contours(hand_masked)
		if contours is not None and len(contours) > 0:
			max_contour = image_analysis.max_contour(contours)

			palm_radius, palm_center = HD.find_palm(max_contour)
			hull = image_analysis.hull(max_contour)
			fingers = HD.find_fingers(hull, palm_center, palm_radius)
			centroid = image_analysis.centroid(max_contour) # currently not use
			defects = image_analysis.defects(max_contour)

			if centroid is not None and defects is not None and len(defects) > 0:
				farthest_point = image_analysis.farthest_point(defects, max_contour, centroid)
				self.buffer_hand(centroid, frame_number, MD)
				if farthest_point is not None:
					self.plot_farthest_point(origin_frame, farthest_point)
					self.plot_hull(origin_frame, hull)
					self.plot_palm_circle(origin_frame, palm_center, palm_radius)
					self.plot_fingers(origin_frame, fingers)

		if self.is_motion(MD):
			win32api.keybd_event(0xB0,0 , 0,0)
			print ("is horizontal slide")
		frame_final = np.vstack([origin_frame, hand_masked])
		return frame_final

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

	def buffer_hand(self, centroid, frame_number, MD):
		# save motion every 1/4 sec, fps 
		MD.save_to_buffer(centroid, frame_number)

	def is_motion(self, MD):
		return MD.match_motion()
 
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