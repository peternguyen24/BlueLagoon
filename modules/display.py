import cv2
import numpy as np
import math
from .frame_processor import FrameProcessor
from .hand_detection import HandDetection
from .background_detection import BackgroundDetection

def loop():

    camera = cv2.VideoCapture(0)

    # modules
    FP = FrameProcessor()
    BD = BackgroundDetection()
    HD = HandDetection()

    while(True):
        #get_frame        
        ret, frame = camera.read()

        #save the origin frame
        origin_frame = frame.copy()

        # resize and flip frame
        frame = FP.resize(frame)
        frame = FP.flip(frame)

        #final_frame 
        final_frame = frame.copy()

        keystroke = cv2.waitKey(30)
        if  keystroke == 27:
            break
        elif keystroke == ord('b'):
            if not BD.trained_background:
                print("Trained background")
                BD.train_background(frame)
        elif keystroke == 32:
            if not HD.trained_hand:
                HD.train_hand(frame)

        if not HD.trained_hand:
            final_frame = HD.draw_hand_rect(final_frame)
        else:
            final_frame = FP.draw_final(final_frame, HD, BD)

        cv2.imshow('my webcam', final_frame)
    cv2.destroyAllWindows()






