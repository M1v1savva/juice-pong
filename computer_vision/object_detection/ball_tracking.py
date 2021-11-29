import os
import sys

sys.path.append(os.path.dirname(os.path.abspath('README.md')) + '/computer_vision/yolov5')

from detect import run


def yolo_run(imgsz, cam_type, weights_path, source):
    run(imgsz=imgsz, cam_type=cam_type, weights=weights_path, source=source)
