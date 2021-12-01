import os
import cv2
import sys
import time
import argparse
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
    for i in range(0):
        time.sleep(2)
        frame = cv2.VideoCapture(source)
        # frame.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # frame.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame = grab_frame(frame)
        print(i)
        cv2.imwrite(f'computer_vision/data/output/frames/new_cam{450+(i)}.png', frame)

def yolo_detect(imgsz, source, cam_type, weights):
    yolo_run(imgsz=imgsz, cam_type=cam_type, weights_path=os.path.dirname(os.path.abspath('README.md')) + f'/computer_vision/weights/{weights}/{imgsz}/last.pt', source=source)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsz', type=int, default=320, help='input image size')
    parser.add_argument('--source', type=int, default=0, help='webcam source')
    parser.add_argument('--cam_type', type=str, default='overhead', help='camera angle type, either overhead or front')
    parser.add_argument('--weights', type=str, default='cam1', help='yolo weights type, either cam1 or cam2')
    return parser.parse_args()

def main(p):
    yolo_detect(**vars(p))

if __name__ == "__main__":
    opt = parse_opt()
    main(p=opt)

    # python3 computer_vision/object_detection/runscript.py --imgsz 320 --source 0 --cam_type 'overhead' --weights 'cam1'
    # python3 computer_vision/object_detection/runscript.py --imgsz 320 --source 1 --cam_type 'front' --weights 'cam2'

    # python3 computer_vision/object_detection/runscript.py --imgsz 320 --source 0 --cam_type 'overhead' --weights 'cam1' & python3 computer_vision/object_detection/runscript.py --imgsz 320 --source 1 --cam_type 'front' --weights 'cam2'


    # save_images(source=1)
    

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


    # import shutil, random

    # source = 'computer_vision/new_frames'
    # target = 'computer_vision/new_frames_shuffled'

    # l = os.listdir(source)
    # random.shuffle(l)
    # i = 242
    # for f in l:
    #     if not f.startswith('.'):
    #         shutil.copyfile(f'{source}/{f}', f'{target}/new_front_cam_{i}.png')
    #         i+=1
