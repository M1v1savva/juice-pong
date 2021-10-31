from computer_vision.yolov5.detect import run


def yolo_run():
    run(weights='weights/last.pt', source=0)
