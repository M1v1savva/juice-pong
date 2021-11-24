import os
import cv2
import sys

sys.path.append(os.path.dirname(os.path.abspath('README.md')) + '/computer_vision')

from ball_tracking import yolo_run


def grab_frame(cap):
    ret, frame = cap.read()
    return frame


def save_images():
    for i in range(1):
        frame = cv2.VideoCapture(0)
        frame.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        frame.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame = grab_frame(frame)
        cv2.imwrite(f'computer_vision/data/output/frames/front_cam{i}.png', frame)


if __name__ == "__main__":
    imgsz = 640
    cam_type = 'overhead'
    source = 0
    yolo_run(imgsz=imgsz, cam_type=cam_type, weights_path=os.path.dirname(os.path.abspath('README.md')) + f'/computer_vision/weights/{cam_type}/640/last.pt', source=source)
    # save_images()
