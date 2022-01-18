# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import math
import os
import sys
from pathlib import Path
# from scipy.spatial import distance as dist

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

sys.path.append(os.path.dirname(os.path.abspath('README.md')) + '/computer_vision/object_detection')

from cup_detection import hough_transform


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def euclidean_dist(ptA, ptB, scale_fact=1):
    if ptA[0] > ptB[0]:
        sign = -1
    else:
        sign = 1
    return (math.sqrt(math.pow(ptA[0] - ptB[0], 2) + math.pow(ptA[1] - ptB[1], 2)) / scale_fact) * sign


def ball_in_cup(cups_center_coordinates_list, ball_center_coordinate, tolerance, ball_size):
    # if ball_size < 50:
    if len(ball_center_coordinate) > 0 and len(cups_center_coordinates_list) > 0:
        for c in cups_center_coordinates_list:
            if ball_center_coordinate[0] in range(c[0] - tolerance, c[0] + tolerance) and ball_center_coordinate[1] in range(c[1] - tolerance, c[1] + tolerance):
                return True, c
    return False, ()


def noisy(image, mean=0, var=0.1):
    row, col, ch= image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    is_human_shot = False
    is_robot_shot = False
    human_shot_counter = 0
    human_cups_scored_counter = 0
    robot_shot_counter = 0
    robot_cups_scored_counter = 0
    ball_speed = 0
    ball_detected_coordinates = ()
    ball_size = sys.maxsize
    saved_time = time_sync()
    turn_state = 0   # 0 for human, 1 for robot
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        if dnn:
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        else:
            check_requirements(('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))
            import onnxruntime
            session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    flag = False
    shot_time = 0
    for path, img, im0s, vid_cap in dataset:
        # img = noisy(img, 0, 0.1)
        t1 = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            if dnn:
                net.setInput(img)
                pred = torch.tensor(net.forward())
            else:
                pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            ball_center_coordinate = ()
            qr_code_center_coordinate_list = []
            qr_code_corner_coordinate_list = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                best_ball = (0, '0', 0)
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        center_coordinate = int((int(xyxy[0]) + int(xyxy[2])) / 2), int((int(xyxy[1]) + int(xyxy[3])) / 2)
                        if label[:-5] == 'ball':
                            # annotator.box_label(xyxy, label, color=colors(c, True))
                            if float(label[-5:]) > 0.5 and float(label[-5:]) > float(best_ball[1][-5:]):
                                best_ball = (xyxy, label, c)
                                ball_center_coordinate = center_coordinate
                                ball_size = euclidean_dist((int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])))
                                # print(ball_size)
                        elif label[:-5] == 'qr_code':
                            if float(label[-5:]) > 0.8:
                                annotator.box_label(xyxy, 'qr' + label[-5:], color=(0, 0, 0))
                                qr_code_center_coordinate_list.append(center_coordinate)
                                qr_code_corner_coordinate_list.append(xyxy)
                        elif label[:-5] == 'human':
                            annotator.box_label(xyxy, label, color=(100, 100, 0))
                            qr_code_center_coordinate_list.append(center_coordinate)
                            qr_code_corner_coordinate_list.append(xyxy)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                if best_ball[1] != '0':

                    annotator.box_label(best_ball[0], best_ball[1], color=colors(best_ball[2], True))

            # Print time (inference-only)
            # print(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()

            if len(qr_code_corner_coordinate_list) == 2:
                cups_center_coordinate = hough_transform(im0)
                D1 = abs(euclidean_dist((int(qr_code_corner_coordinate_list[0][0]), (int(qr_code_corner_coordinate_list[0][1]))), (int(qr_code_corner_coordinate_list[0][2]), int(qr_code_corner_coordinate_list[0][3])), 1))
                D2 = abs(euclidean_dist((int(qr_code_corner_coordinate_list[1][0]), (int(qr_code_corner_coordinate_list[1][1]))), (int(qr_code_corner_coordinate_list[1][2]), int(qr_code_corner_coordinate_list[1][3])), 1))

                robot_cups_real_distance_list = []
                human_cups_real_distance_list = []
                robot_cups_center_coordinate = []
                human_cups_center_coordinate = []
                for c in cups_center_coordinate:
                    dist_from_qr_1 = euclidean_dist((c[0], c[1]), (qr_code_center_coordinate_list[0][0], qr_code_center_coordinate_list[0][1]), D1 / 12)
                    dist_from_qr_2 = euclidean_dist((c[0], c[1]), (qr_code_center_coordinate_list[1][0], qr_code_center_coordinate_list[1][1]), D2 / 12)
                    if dist_from_qr_1 > 0 and dist_from_qr_2 > 0:
                        human_cups_real_distance_list.append((dist_from_qr_1, dist_from_qr_2))
                        human_cups_center_coordinate.append(c)
                        cv2.line(im0, (c[0], c[1]), (qr_code_center_coordinate_list[0][0], qr_code_center_coordinate_list[0][1]), (0, 215, 255), 2)
                        cv2.line(im0, (c[0], c[1]), (qr_code_center_coordinate_list[1][0], qr_code_center_coordinate_list[1][1]), (0, 215, 255), 2)
                    else:
                        robot_cups_real_distance_list.append((dist_from_qr_1, dist_from_qr_2))
                        robot_cups_center_coordinate.append(c)

                for idx in range(len(human_cups_real_distance_list)):
                    text1_X, text1_Y = midpoint((human_cups_center_coordinate[idx][0], human_cups_center_coordinate[idx][1]), (qr_code_center_coordinate_list[0][0], qr_code_center_coordinate_list[0][1]))
                    text2_X, text2_Y = midpoint((human_cups_center_coordinate[idx][0], human_cups_center_coordinate[idx][1]), (qr_code_center_coordinate_list[1][0], qr_code_center_coordinate_list[1][1]))
                    cv2.putText(im0, str(round(human_cups_real_distance_list[idx][0], 2)), (int(text1_X), int(text1_Y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
                    cv2.putText(im0, str(round(human_cups_real_distance_list[idx][1], 2)), (int(text2_X), int(text2_Y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
                
                for coord in human_cups_center_coordinate:
                    cv2.rectangle(im0, (coord[0]-5, coord[1]-5), (coord[0]+5, coord[1]+5), (0, 0, 255), -1)

                for coord in robot_cups_center_coordinate:
                    cv2.rectangle(im0, (coord[0]-5, coord[1]-5), (coord[0]+5, coord[1]+5), (255, 0, 0), -1)

                if len(ball_center_coordinate) > 0:
                    dist_from_qr_1 = euclidean_dist((ball_center_coordinate[0], ball_center_coordinate[1]), (qr_code_center_coordinate_list[0][0], qr_code_center_coordinate_list[0][1]), D1 / 12)
                    dist_from_qr_2 = euclidean_dist((ball_center_coordinate[0], ball_center_coordinate[1]), (qr_code_center_coordinate_list[1][0], qr_code_center_coordinate_list[1][1]), D2 / 12)
                    ball_real_distance = (dist_from_qr_1, dist_from_qr_2)
                    cv2.line(im0, (ball_center_coordinate[0], ball_center_coordinate[1]), (qr_code_center_coordinate_list[0][0], qr_code_center_coordinate_list[0][1]), (215, 255, 0), 2)
                    cv2.line(im0, (ball_center_coordinate[0], ball_center_coordinate[1]), (qr_code_center_coordinate_list[1][0], qr_code_center_coordinate_list[1][1]), (215, 255, 0), 2)

                    text1_X, text1_Y = midpoint((ball_center_coordinate[0], ball_center_coordinate[1]), (qr_code_center_coordinate_list[0][0], qr_code_center_coordinate_list[0][1]))
                    text2_X, text2_Y = midpoint((ball_center_coordinate[0], ball_center_coordinate[1]), (qr_code_center_coordinate_list[1][0], qr_code_center_coordinate_list[1][1]))
                    cv2.putText(im0, str(round(ball_real_distance[0], 2)), (int(text1_X), int(text1_Y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 255, 0), 2)
                    cv2.putText(im0, str(round(ball_real_distance[1], 2)), (int(text2_X), int(text2_Y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 255, 0), 2)

                    min_speed_limit = 15   # in km/h
                    max_speed_limit = 80   # in km/h
                    time_limit = 2    # in sec
                    if len(ball_detected_coordinates) > 0:
                        real_distance_btw_balls = euclidean_dist((ball_detected_coordinates[0], ball_detected_coordinates[1]), (ball_center_coordinate[0], ball_center_coordinate[1]), ((D1+D2)/2)/12)
                        ball_speed = (real_distance_btw_balls/100000)/((time_sync()-t4)/3600)
                        delay_between_shots = 10   # in sec
                        if time_sync()-shot_time > delay_between_shots:
                            if turn_state == 0 and ball_speed > min_speed_limit and ball_speed < max_speed_limit and time_sync()-saved_time > time_limit:
                                human_shot_counter += 1
                                saved_time = time_sync()
                                is_human_shot = True
                                turn_state = 1
                                shot_time = time_sync()
                            elif turn_state == 1 and ball_speed < (min_speed_limit*-1) and ball_speed  > (max_speed_limit*-1) and time_sync()-saved_time > time_limit:
                                robot_shot_counter += 1
                                saved_time = time_sync()
                                is_robot_shot = True
                                turn_state = 0
                                shot_time = time_sync()
                    else:
                        ball_speed = 0
                    ball_detected_coordinates = (ball_center_coordinate[0], ball_center_coordinate[1])

                    min_shot_delay_tolerance = 1    # in sec
                    max_shot_delay_tolerance = 2    # in sec
                    if (is_human_shot or is_robot_shot) and time_sync()-saved_time > max_shot_delay_tolerance:
                        is_human_shot = False
                        is_robot_shot = False
                    flag, coordinate = ball_in_cup(cups_center_coordinates_list=cups_center_coordinate, ball_center_coordinate=ball_center_coordinate, tolerance=40, ball_size=ball_size)
                    if flag:
                        cv2.circle(im0, (coordinate[0], coordinate[1]), 50, (0, 255, 0), 4)
                        if is_human_shot and time_sync()-saved_time > min_shot_delay_tolerance and time_sync()-saved_time < max_shot_delay_tolerance:
                            is_human_shot = False
                            human_cups_scored_counter += 1
                        elif is_robot_shot and time_sync()-saved_time > min_shot_delay_tolerance and time_sync()-saved_time < max_shot_delay_tolerance:
                            is_robot_shot = False
                            robot_cups_scored_counter += 1
                else:
                    ball_speed = 0
                    ball_detected_coordinates = ()

                # human number of cups left
                cv2.putText(im0, f'# Human cups: {str(len(human_cups_center_coordinate))}', (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                # robot number of cups left
                cv2.putText(im0, f'# Robot cups: {str(len(robot_cups_center_coordinate))}', (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                cv2.putText(im0, f'Human (scored | shot): ({human_cups_scored_counter} | {human_shot_counter})', (60, 605), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(im0, f'Robot (scored | shot): ({robot_cups_scored_counter} | {robot_shot_counter})', (60, 655), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                if abs(ball_speed) < 0.2:
                    ball_speed = 0
                cv2.putText(im0, f'Ball speed (km/h): {abs(round(ball_speed, 2))}', (60, 705), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                if time_sync()-shot_time < 10:
                    cv2.putText(im0, f'Timer: {round((10-(time_sync()-shot_time)), 1)}', (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                else:
                    cv2.putText(im0, f'Ready', (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 175, 0), 3)
            else:
                if len(qr_code_corner_coordinate_list) < 2:
                    cv2.putText(im0, 'Error: please make free spaces around qr codes', (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                elif len(qr_code_corner_coordinate_list) > 2:
                    cv2.putText(im0, 'Error: More than 2 qr codes are detected', (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        
            t4 = time_sync()
            # print(f'{int(1/(t4 - t1))} fps')
            frame_rate = 1/(t4 - t1)
            if frame_rate > 1:
                cv2.putText(im0, f'{int(frame_rate)} fps', (1100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            else:
                cv2.putText(im0, f'{round(frame_rate, 2)} fps', (1100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            
            if turn_state == 0:
                cv2.putText(im0, f'Human to play', (1000, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            elif turn_state == 1:
                cv2.putText(im0, f'Robot to play', (1000, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
