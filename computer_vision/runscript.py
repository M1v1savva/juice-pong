import os
import sys
import cv2
import argparse

sys.path.append(os.path.dirname(os.path.abspath('README.md')) + '/computer_vision')

from yolov5.detect import run


def grab_frame(cap):
    ret, frame = cap.read()
    return frame

def save_images(source):
    for i in range(1):
        # time.sleep(2)
        frame = cv2.VideoCapture(source)
        frame.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        frame.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame = grab_frame(frame)
        print(i)
        cv2.imwrite(f'computer_vision/data/output/frames/new_swarm_cam{100+(i)}.png', frame)

def yolo_detect(imgsz, source):
    run(imgsz=imgsz, source=source, weights=f'computer_vision/yolov5/weights/{imgsz}/best.pt')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsz', type=int, default=160, help='input image size')
    parser.add_argument('--source', type=int, default=0, help='webcam source')
    return parser.parse_args()

def main(p):
    yolo_detect(**vars(p))

if __name__ == "__main__":


    opt = parse_opt()
    main(p=opt)

    # python3 computer_vision/runscript.py --source 0
