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
    parser.add_argument('--imgsz', type=int, default=320, help='input image size')
    parser.add_argument('--source', type=int, default=0, help='webcam source')
    return parser.parse_args()

def main(p):
    yolo_detect(**vars(p))

if __name__ == "__main__":


    opt = parse_opt()
    main(p=opt)


    # python3 computer_vision/runscript.py --imgsz 160 --source 0

    # cd computer_vision/sparseml/integrations/ultralytics-yolov5/yolov5
    # python3 detect.py --weights weights/pruned/640/best.pt --img-size 640 --source 0


    # save_images(source=0)
    
    
    # fps = 60
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    # cap.set(cv2.CAP_PROP_FPS, fps)
    # # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

    # print(cap.get(cv2.CAP_PROP_FPS))

    # # Check if the webcam is opened correctly
    # if not cap.isOpened():
    #     raise IOError("Cannot open webcam")

    # i = 0
    # stop = 100
    # while True:
    #     ret, frame = cap.read()
    #     # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    #     # frame = grab_frame(frame)
    #     print(i)
    #     time.sleep(0.5)
    #     cv2.imwrite(f'computer_vision/data/output/frames/new_swarm_cam{i+550}.png', frame)
    #     i+=1
    #     if i == stop:
    #         break
    #     # cv2.imshow('image', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()


    # import shutil, random

    # source = 'computer_vision/data/output/frames'
    # target = 'computer_vision/data/output/shuffled_frames'

    # l = os.listdir(source)
    # random.shuffle(l)
    # i = 0
    # for f in l:
    #     if not f.startswith('.'):
    #         shutil.copyfile(f'{source}/{f}', f'{target}/new_swarm_cam{i}.png')
    #         i+=1