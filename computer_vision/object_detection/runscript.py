import os
import cv2
import sys
import time
import matplotlib.pyplot as plt
import threading

sys.path.append(os.path.dirname(os.path.abspath('README.md')) + '/computer_vision')

from multiprocessing import Process
from ball_tracking import yolo_run


class yoloThread(threading.Thread):
    def __init__(self, imgsz, cam_type, source):
        threading.Thread.__init__(self)
        self.imgsz = imgsz
        self.cam_type = cam_type
        self.source = source
    def run(self):
        yolo_run(imgsz=self.imgsz, cam_type=self.cam_type, weights_path=os.path.dirname(os.path.abspath('README.md')) + f'/computer_vision/weights/overhead/{self.imgsz}/last.pt', source=self.source)


def grab_frame(cap):
    ret, frame = cap.read()
    return frame


def save_images(source):
    for i in range(1):
        # time.sleep(1)
        frame = cv2.VideoCapture(source)
        # frame.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # frame.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame = grab_frame(frame)
        print(i)
        cv2.imwrite(f'computer_vision/data/output/frames/new_cam{400+(i)}.png', frame)

def yolo_detect(imgsz, source, cam_type, weights):
    yolo_run(imgsz=imgsz, cam_type=cam_type, weights_path=os.path.dirname(os.path.abspath('README.md')) + f'/computer_vision/weights/{weights}/{imgsz}/last.pt', source=source)


if __name__ == "__main__":
    # save_images(source=1)

    # yolo_detect(160, source=1, cam_type='front', weigths='cam1_cam2')
    # yolo_detect(640, source=0, cam_type='overhead', weigths='cam1_cam2')

    yolo_detect(320, source=0, cam_type='front', weights='cam2')
    
    # yolo_thread_overhead = yoloThread(imgsz=320, cam_type='overhead', source=0)
    # yolo_thread_front = yoloThread(imgsz=320, cam_type='front', source=2)
    # yolo_thread_overhead.start()
    # yolo_thread_front.start()
    

    # import cv2

    # cap = cv2.VideoCapture(0)

    # # Check if the webcam is opened correctly
    # if not cap.isOpened():
    #     raise IOError("Cannot open webcam")

    # while True:
    #     ret, frame = cap.read()
    #     # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    #     cv2.imshow('Input', frame)

    #     c = cv2.waitKey(1)
    #     if c == 27:
    #         break

    # cap.release()
    # cv2.destroyAllWindows()
