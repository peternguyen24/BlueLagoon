import cv2
import numpy as np
import math
import time
from .frame_processor import FrameProcessor
from .hand_detection import HandDetection
from .background_detection import BackgroundDetection
from .movement_detector import MovementDetector

def estimate_fps(video_cam):
    counter = 0
    start = time.time()
    while(True):
        ret, frame = video_cam.read()
        end = time.time()
        counter += 1
        fps = counter/(end - start)

        if counter > 150 and counter % 15 == 0:
            print(fps)

def loop():

    camera = cv2.VideoCapture(0)
    counter = 0

    # modules
    FP = FrameProcessor()
    BD = BackgroundDetection()
    HD = HandDetection()
    MD = MovementDetector()

    while(True):
        #start calc fps (at frame 150, sec 5 if fps is 30)
        if counter == 10:
            start = time.time()

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
                BD.set_static_background(frame)
        elif keystroke == 32:
            if not HD.trained_hand:
                HD.train_hand(frame)
                start = time.time()
                counter = 0

        if not HD.trained_hand:
            final_frame = HD.draw_hand_rect(final_frame)
        else:
            final_frame = FP.draw_final(final_frame, HD, BD, MD, counter)

        # end calc fps after
        counter+=1
        if counter == 50:
            end = time.time()
            fps = counter/(end-start)
            MD.set_fps(fps)
            print("fps set to: ", fps)

        # reset frame counter for motion detection
        if counter > 300:
            counter = 0

        cv2.imshow('my webcam', final_frame)
    cv2.destroyAllWindows()






