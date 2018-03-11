import cv2
import numpy as np

class HandDetection:
	def __init__(self):
		self.trained_hand = False
		self.hand_row_nw = None
		self.hand_row_se = None
		self.hand_col_nw = None
		self.hand_col_se = None
		self.hand_hist = None

	def draw_hand_rect(self, frame):  
	    rows,cols,_ = frame.shape

	    self.hand_row_nw = np.array([6*rows//20,6*rows//20,6*rows//20,10*rows//20,10*rows//20,10*rows//20,14*rows//20,14*rows//20,14*rows//20])

	    self.hand_col_nw = np.array([9*cols//20,10*cols//20,11*cols//20,9*cols//20,10*cols//20,11*cols//20,9*cols//20,10*cols//20,11*cols//20])

	    self.hand_row_se = self.hand_row_nw + 10
	    self.hand_col_se = self.hand_col_nw + 10

	    size = self.hand_row_nw.size
	    for i in range(size):
	        cv2.rectangle(frame,(self.hand_col_nw[i],self.hand_row_nw[i]),(self.hand_col_se[i],self.hand_row_se[i]),(0,255,0),1)
	    frame_final = frame
	    return frame_final

	def train_hand(self, frame):
		self.trained_hand = True
		self.set_hand_hist(frame)

	def set_hand_hist(self, frame):
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		roi = np.zeros([90,10,3], dtype=hsv.dtype)

		size = self.hand_row_nw.size
		for i in range(size):
			roi[i*10:i*10+10,0:10] = hsv[self.hand_row_nw[i]:self.hand_row_nw[i]+10, self.hand_col_nw[i]:self.hand_col_nw[i]+10]
		self.hand_hist = cv2.calcHist([roi],[0, 1], None, [180, 256], [0, 180, 0, 256])
		cv2.normalize(self.hand_hist, self.hand_hist, 0, 255, cv2.NORM_MINMAX)
	def find_palm(self, contour):
		x,y,w,h =cv2.boundingRect(contour) #rectangle cover the contour
		max_d = 0
		pt = None
		# Divide rectangle to 100x100 and check around 10000 point.
		for ind_y in range(int(y+0.2*h),int(y+0.8*h),max(1, int(h*0.6/100))): #around 0.25 to 0.6 region of height (Faster calculation with ok results)
			for ind_x in range(int(x+0.3*w),int(x+0.7*w), max(1, int(h*0.4/100))): #around 0.3 to 0.6 region of width (Faster calculation with ok results)
				dist= cv2.pointPolygonTest(contour,(ind_x,ind_y),True)
				if(dist>max_d):
					max_d=int(dist)
					pt=(ind_x,ind_y)
		return max_d, pt
	def find_fingers(self, hull, palm_center, palm_radius):
		finger_thresh_l = 2.0
		finger_thresh_u = 3.8
		fingers = []
		hullpoints = []
		# Only take 1 point per cluster. cluster distance = 20 (^2=400)		
		for i in range(len(hull)):
			dist2 = (hull[-i][0][0] - hull[-i+1][0][0])**2 + (hull[-i][0][1] - hull[-i+1][0][1])**2
			if (dist2 > 400):
				hullpoints.append((hull[-i][0][0],hull[-i][0][1]))
		for i in range(len(hullpoints)):
			dist = np.sqrt((hullpoints[i][0] - palm_center[0])**2 + (hullpoints[i][1] - palm_center[1])**2)
			if (dist>finger_thresh_l*palm_radius and 
				dist<finger_thresh_u*palm_radius and
				hullpoints[i][1] < palm_center[1] + palm_radius):
				fingers.append(hullpoints[i])
		return fingers	
    