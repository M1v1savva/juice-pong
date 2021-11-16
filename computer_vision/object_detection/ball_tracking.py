from computer_vision.yolov5.detect import run


def yolo_run():
    run(weights='weights/best/last.pt', source=0)
