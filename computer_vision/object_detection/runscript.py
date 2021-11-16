import cv2

from computer_vision.object_detection.ball_tracking import yolo_run


def grab_frame(cap):
    ret, frame = cap.read()
    return frame


def save_images():
    for i in range(50):
        frame = cv2.VideoCapture(0)
        frame = grab_frame(frame)
        cv2.imwrite(f'data/output/frames/qr_pic{i+200}.png', frame)


if __name__ == "__main__":
    yolo_run()
    # save_images()
