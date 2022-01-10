# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Annotation script for running YOLO models using DeepSparse and other inference engines.
Supports .jpg images, .mp4 movies, and webcam streaming.

##########
Command help:
usage: annotate.py [-h] --source SOURCE [-e {deepsparse,onnxruntime,torch}]
                   [--image-shape IMAGE_SHAPE [IMAGE_SHAPE ...]]
                   [-c NUM_CORES] [-q] [--fp16]
                   [--device DEVICE] [--save-dir SAVE_DIR] [--name NAME]
                   [--target-fps TARGET_FPS] [--no-save]
                   [--model-config MODEL_CONFIG]
                   model_filepath

Annotate images, videos, and streams with sparsified or non-sparsified YOLO models

positional arguments:
  model_filepath        The full file path of the ONNX model file or SparseZoo
                        stub to the model for DeepSparse and ONNX Runtime
                        Engines. Path to a .pt loadable PyTorch Module for
                        torch - the Module can be the top-level object loaded
                        or loaded into 'model' in a state dict

optional arguments:
  -h, --help            show this help message and exit
  --source SOURCE       File path to image or directory of .jpg files, a .mp4
                        video, or an integer (i.e. 0) for webcam
  -e {deepsparse,onnxruntime,torch}, --engine {deepsparse,onnxruntime,torch}
                        Inference engine backend to run on. Choices are
                        'deepsparse', 'onnxruntime', and 'torch'. Default is
                        'deepsparse'
  --image-shape IMAGE_SHAPE [IMAGE_SHAPE ...]
                        Image shape to run model with, must be two integers.
                        Default is 640 640
  -c NUM_CORES, --num-cores NUM_CORES
                        The number of physical cores to run the annotations
                        with, defaults to None where it uses all physical
                        cores available on the system. For DeepSparse
                        benchmarks, this value is the number of cores per
                        socket
  -q, --quantized-inputs
                        Set flag to execute with int8 inputs instead of
                        float32
  --fp16                Set flag to execute with torch in half precision
                        (fp16)
  --device DEVICE       Torch device id to run the model with. Default is cpu.
                        Non-cpu only supported for Torch benchmarking. Default
                        is 'cpu' unless running with Torch and CUDA is
                        available, then cuda on device 0. i.e. 'cuda', 'cpu',
                        0, 'cuda:1'
  --save-dir SAVE_DIR   directory to save all results to. defaults to
                        'annotation_results'
  --name NAME           name of directory in save-dir to write results to.
                        defaults to {engine}-annotations-{run_number}
  --target-fps TARGET_FPS
                        target FPS when writing video files. Frames will be
                        dropped to closely match target FPS. --source must be
                        a video file and if target-fps is greater than the
                        source video fps then it will be ignored. Default is
                        None
  --no-save             set flag when source is from webcam to not save
                        results. not supported for non-webcam sources
  --model-config MODEL_CONFIG
                        YOLO config YAML file to override default anchor
                        points when post-processing. Defaults to use standard
                        YOLOv3/YOLOv5 anchors

##########
Example command for running webcam annotations with pruned quantized YOLOv3:
python annotate.py \
    zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned_quant-aggressive_94 \
    --source 0 \
    --quantized-inputs \
    --image-shape 416 416

##########
Example command for running video annotations with pruned YOLOv5l:
python annotate.py \
    zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned-aggressive_98 \
    --source my_video.mp4 \
    --image-shape 416 416

##########
Example command for running image annotations with using PyTorch CPU YOLOv3:
python annotate.py \
    path/to/yolo-v3.pt \
    --source path/to/my/jpg/images \
    --device cpu \
    --image-shape 416 416
"""


import argparse
import itertools
import logging
import sys
import os
import time
from typing import Any, List, Union

import math
import numpy as np
import onnx
import onnxruntime

import cv2
import torch
from deepsparse import compile_model
from deepsparse_utils import (
    YoloPostprocessor,
    get_yolo_loader_and_saver,
    modify_yolo_onnx_input_shape,
    postprocess_nms,
    yolo_onnx_has_postprocessing,
)
from sparseml.onnx.utils import override_model_batch_size
from typing import Any, List, Union


DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"
TORCH_ENGINE = "torch"

_LOGGER = logging.getLogger(__name__)


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(
        description="Annotate images, videos, and streams with sparsified YOLO models"
    )

    parser.add_argument(
        "model_filepath",
        type=str,
        help=(
            "The full file path of the ONNX model file or SparseZoo stub to the model "
            "for DeepSparse and ONNX Runtime Engines. Path to a .pt loadable PyTorch "
            "Module for torch - the Module can be the top-level object "
            "loaded or loaded into 'model' in a state dict"
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help=(
            "File path to image or directory of .jpg files, a .mp4 video, "
            "or an integer (i.e. 0) for webcam"
        ),
    )
    parser.add_argument(
        "-e",
        "--engine",
        type=str,
        default=DEEPSPARSE_ENGINE,
        choices=[DEEPSPARSE_ENGINE, ORT_ENGINE, TORCH_ENGINE],
        help=(
            "Inference engine backend to run on. Choices are 'deepsparse', "
            "'onnxruntime', and 'torch'. Default is 'deepsparse'"
        ),
    )
    parser.add_argument(
        "--image-shape",
        type=int,
        default=(640, 640),
        nargs="+",
        help="Image shape to run model with, must be two integers. Default is 640 640",
    )
    parser.add_argument(
        "-c",
        "--num-cores",
        type=int,
        default=None,
        help=(
            "The number of physical cores to run the annotations with, "
            "defaults to None where it uses all physical cores available on the system."
            " For DeepSparse benchmarks, this value is the number of cores per socket"
        ),
    )
    parser.add_argument(
        "-q",
        "--quantized-inputs",
        help=("Set flag to execute with int8 inputs instead of float32"),
        action="store_true",
    )
    parser.add_argument(
        "--fp16",
        help=("Set flag to execute with torch in half precision (fp16)"),
        action="store_true",
    )
    parser.add_argument(
        "--device",
        type=_parse_device,
        default=None,
        help=(
            "Torch device id to run the model with. Default is cpu. Non-cpu "
            " only supported for Torch benchmarking. Default is 'cpu' "
            "unless running with Torch and CUDA is available, then CUDA on "
            "device 0. i.e. 'cuda', 'cpu', 0, 'cuda:1'"
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="annotation_results",
        help="directory to save all results to. defaults to 'annotation_results'",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "name of directory in save-dir to write results to. defaults to "
            "{engine}-annotations-{run_number}"
        ),
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help=(
            "target FPS when writing video files. Frames will be dropped to "
            "closely match target FPS. --source must be a video file and if target-fps "
            "is greater than the source video fps then it will be ignored. Default is "
            "None"
        ),
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help=(
            "set flag when source is from webcam to not save results. not supported "
            "for non-webcam sources"
        ),
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help=(
            "YOLO config YAML file to override default anchor points when "
            "post-processing. Defaults to use standard YOLOv3/YOLOv5 anchors"
        ),
    )

    args = parser.parse_args(args=arguments)
    if args.engine == TORCH_ENGINE and args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    return args


def _parse_device(device: Union[str, int]) -> Union[str, int]:
    try:
        return int(device)
    except Exception:
        return device


def _get_save_dir(args) -> str:
    name = args.name or f"{args.engine}-annotations"
    save_dir = os.path.join(args.save_dir, name)
    idx = 2
    while os.path.exists(save_dir):
        save_dir = os.path.join(args.save_dir, f"{name}-{idx}")
        idx += 1
    _LOGGER.info(f"Results will be saved to {save_dir}")
    return save_dir


def _load_model(args) -> Any:
    # validation
    if args.device not in [None, "cpu"] and args.engine != TORCH_ENGINE:
        raise ValueError(f"device {args.device} is not supported for {args.engine}")
    if args.fp16 and args.engine != TORCH_ENGINE:
        raise ValueError(f"half precision is not supported for {args.engine}")
    if args.quantized_inputs and args.engine == TORCH_ENGINE:
        raise ValueError(f"quantized inputs not supported for {args.engine}")
    if args.num_cores is not None and args.engine == TORCH_ENGINE:
        raise ValueError(
            f"overriding default num_cores not supported for {args.engine}"
        )
    if (
        args.num_cores is not None
        and args.engine == ORT_ENGINE
        and onnxruntime.__version__ < "1.7"
    ):
        raise ValueError(
            "overriding default num_cores not supported for onnxruntime < 1.7.0. "
            "If using an older build with OpenMP, try setting the OMP_NUM_THREADS "
            "environment variable"
        )

    # scale static ONNX graph to desired image shape
    if args.engine in [DEEPSPARSE_ENGINE, ORT_ENGINE]:
        args.model_filepath, _ = modify_yolo_onnx_input_shape(
            args.model_filepath, args.image_shape
        )
        has_postprocessing = yolo_onnx_has_postprocessing(args.model_filepath)

    # load model
    if args.engine == DEEPSPARSE_ENGINE:
        _LOGGER.info(f"Compiling DeepSparse model for {args.model_filepath}")
        model = compile_model(args.model_filepath, 1, args.num_cores)
        if args.quantized_inputs and not model.cpu_vnni:
            _LOGGER.warning(
                "WARNING: VNNI instructions not detected, "
                "quantization speedup not well supported"
            )
    elif args.engine == ORT_ENGINE:
        _LOGGER.info(f"Loading onnxruntime model for {args.model_filepath}")

        sess_options = onnxruntime.SessionOptions()
        if args.num_cores is not None:
            sess_options.intra_op_num_threads = args.num_cores
        sess_options.log_severity_level = 3
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        onnx_model = onnx.load(args.model_filepath)
        override_model_batch_size(onnx_model, 1)
        model = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), sess_options=sess_options
        )
    elif args.engine == TORCH_ENGINE:
        _LOGGER.info(f"Loading torch model for {args.model_filepath}")
        model = torch.load(args.model_filepath)
        if isinstance(model, dict):
            model = model["model"]
        model.to(args.device)
        model.eval()
        if args.fp16:
            _LOGGER.info("Using half precision")
            model.half()
        else:
            _LOGGER.info("Using full precision")
            model.float()
        has_postprocessing = True

    return model, has_postprocessing


def _preprocess_batch(args, batch: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
    if len(batch.shape) == 3:
        batch = batch.reshape(1, *batch.shape)
    if args.engine == TORCH_ENGINE:
        batch = torch.from_numpy(batch.copy())
        batch = batch.to(args.device)
        batch = batch.half() if args.fp16 else batch.float()
        batch /= 255.0
    else:
        if not args.quantized_inputs:
            batch = batch.astype(np.float32) / 255.0
        batch = np.ascontiguousarray(batch)
    return batch


def _run_model(
    args, model: Any, batch: Union[np.ndarray, torch.Tensor]
) -> List[Union[np.ndarray, torch.Tensor]]:
    outputs = None
    if args.engine == TORCH_ENGINE:
        outputs = model(batch)
    elif args.engine == ORT_ENGINE:
        outputs = model.run(
            [out.name for out in model.get_outputs()],  # outputs
            {model.get_inputs()[0].name: batch},  # inputs dict
        )
    else:  # deepsparse
        outputs = model.run([batch])
    return outputs


def annotate(args):
    save_dir = _get_save_dir(args)
    model, has_postprocessing = _load_model(args)
    loader, saver, is_video = get_yolo_loader_and_saver(
        args.source, save_dir, args.image_shape, args
    )
    is_webcam = args.source.isnumeric()

    postprocessor = (
        YoloPostprocessor(args.image_shape, args.model_config)
        if not has_postprocessing
        else None
    )

    is_human_shot = False
    human_shot_counter = 0
    human_cups_scored_counter = 0
    ball_speed = 0
    ball_detected_coordinates = ()
    ball_size = sys.maxsize
    saved_time = time.time()

    score_threshold = 0.5

    for iteration, (inp, source_img) in enumerate(loader):
        if args.device not in ["cpu", None]:
            torch.cuda.synchronize()
        iter_start = time.time()

        # pre-processing
        batch = _preprocess_batch(args, inp)

        # inference
        outputs = _run_model(args, model, batch)

        # post-processing
        if postprocessor:
            outputs = postprocessor.pre_nms_postprocess(outputs)
        else:
            outputs = outputs[0]  # post-processed values stored in first output

        # NMS
        outputs = postprocess_nms(outputs)[0]

        if args.device not in ["cpu", None]:
            torch.cuda.synchronize()

        # annotate
        measured_fps = (
            args.target_fps or (1.0 / (time.time() - iter_start)) if is_video else None
        )

        img_res = np.copy(source_img)

        boxes = outputs[:, 0:4]
        scores = outputs[:, 4]
        labels = outputs[:, 5].astype(int)

        scale_y = source_img.shape[0] / (1.0 * args.image_shape[0]) if args.image_shape else 1.0
        scale_x = source_img.shape[1] / (1.0 * args.image_shape[1]) if args.image_shape else 1.0

        ball_center_coordinate = ()
        qr_code_center_coordinate_list = []
        qr_code_corner_coordinate_list = []

        best_ball_score = 0
        best_ball_idx = 0
        for idx in range(len(labels)):
            if labels[idx] == 0:
                if scores[idx] > best_ball_score:
                    best_ball_score = scores[idx]
                    best_ball_idx = idx

        for idx in range(boxes.shape[0]):
            label = labels[idx].item()
            if scores[idx] > score_threshold:
                annotation_text = (
                    f"{_YOLO_CLASSES[label]}: {scores[idx]:.0%}"
                    if 0 <= label < len(_YOLO_CLASSES)
                    else f"{scores[idx]:.0%}"
                )

                # bounding box points
                left = boxes[idx][0] * scale_x
                top = boxes[idx][1] * scale_y
                right = boxes[idx][2] * scale_x
                bottom = boxes[idx][3] * scale_y

                center_coordinate = int((int(left) + int(right)) / 2), int((int(top) + int(bottom)) / 2)

                if label == 0:
                    if idx == best_ball_idx:
                        ball_center_coordinate = center_coordinate
                        ball_size = euclidean_dist((int(left), int(top)), (int(right), int(bottom)))
                    else:
                        continue
                
                if label == 1:
                    qr_code_center_coordinate_list.append(center_coordinate)
                    qr_code_corner_coordinate_list.append((left, top, right, bottom))

                # calculate text size
                (text_width, text_height), text_baseline = cv2.getTextSize(
                    annotation_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,  # font scale
                    2,  # thickness
                )
                text_height += text_baseline

                # make solid background for annotation text
                cv2.rectangle(
                    img_res,
                    (int(left), int(top) - 33),
                    (int(left) + text_width, int(top) - 28 + text_height),
                    _YOLO_CLASS_COLORS[label],
                    thickness=-1,  # filled solid
                )

                # add white annotation text
                cv2.putText(
                    img_res,
                    annotation_text,
                    (int(left), int(top) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,  # font scale
                    (255, 255, 255),  # white text
                    2,  # thickness
                    cv2.LINE_AA,
                )

                # draw bounding box
                cv2.rectangle(
                    img_res,
                    (int(left), int(top)),
                    (int(right), int(bottom)),
                    _YOLO_CLASS_COLORS[label],
                    thickness=2,
                )

        if measured_fps is not None:
            cv2.putText(img_res, f'{int(measured_fps)} fps', (1100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        annotated_img = img_res

        if len(qr_code_corner_coordinate_list) == 2:
            cups_center_coordinate = hough_transform(annotated_img)
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
                    cv2.line(annotated_img, (c[0], c[1]), (qr_code_center_coordinate_list[0][0], qr_code_center_coordinate_list[0][1]), (0, 215, 255), 2)
                    cv2.line(annotated_img, (c[0], c[1]), (qr_code_center_coordinate_list[1][0], qr_code_center_coordinate_list[1][1]), (0, 215, 255), 2)
                else:
                    robot_cups_real_distance_list.append((dist_from_qr_1, dist_from_qr_2))
                    robot_cups_center_coordinate.append(c)


            for idx in range(len(human_cups_real_distance_list)):
                text1_X, text1_Y = midpoint((human_cups_center_coordinate[idx][0], human_cups_center_coordinate[idx][1]), (qr_code_center_coordinate_list[0][0], qr_code_center_coordinate_list[0][1]))
                text2_X, text2_Y = midpoint((human_cups_center_coordinate[idx][0], human_cups_center_coordinate[idx][1]), (qr_code_center_coordinate_list[1][0], qr_code_center_coordinate_list[1][1]))
                cv2.putText(annotated_img, str(round(human_cups_real_distance_list[idx][0], 2)), (int(text1_X), int(text1_Y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
                cv2.putText(annotated_img, str(round(human_cups_real_distance_list[idx][1], 2)), (int(text2_X), int(text2_Y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
            
            for coord in human_cups_center_coordinate:
                cv2.rectangle(annotated_img, (coord[0]-5, coord[1]-5), (coord[0]+5, coord[1]+5), (0, 0, 255), -1)

            for coord in robot_cups_center_coordinate:
                cv2.rectangle(annotated_img, (coord[0]-5, coord[1]-5), (coord[0]+5, coord[1]+5), (255, 0, 0), -1)

            if len(ball_center_coordinate) > 0:
                dist_from_qr_1 = euclidean_dist((ball_center_coordinate[0], ball_center_coordinate[1]), (qr_code_center_coordinate_list[0][0], qr_code_center_coordinate_list[0][1]), D1 / 12)
                dist_from_qr_2 = euclidean_dist((ball_center_coordinate[0], ball_center_coordinate[1]), (qr_code_center_coordinate_list[1][0], qr_code_center_coordinate_list[1][1]), D2 / 12)
                ball_real_distance = (dist_from_qr_1, dist_from_qr_2)
                cv2.line(annotated_img, (ball_center_coordinate[0], ball_center_coordinate[1]), (qr_code_center_coordinate_list[0][0], qr_code_center_coordinate_list[0][1]), (215, 255, 0), 2)
                cv2.line(annotated_img, (ball_center_coordinate[0], ball_center_coordinate[1]), (qr_code_center_coordinate_list[1][0], qr_code_center_coordinate_list[1][1]), (215, 255, 0), 2)

                text1_X, text1_Y = midpoint((ball_center_coordinate[0], ball_center_coordinate[1]), (qr_code_center_coordinate_list[0][0], qr_code_center_coordinate_list[0][1]))
                text2_X, text2_Y = midpoint((ball_center_coordinate[0], ball_center_coordinate[1]), (qr_code_center_coordinate_list[1][0], qr_code_center_coordinate_list[1][1]))
                cv2.putText(annotated_img, str(round(ball_real_distance[0], 2)), (int(text1_X), int(text1_Y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 255, 0), 2)
                cv2.putText(annotated_img, str(round(ball_real_distance[1], 2)), (int(text2_X), int(text2_Y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 255, 0), 2)

                speed_limit = 10   # in km/h
                time_limit = 3    # in sec
                if len(ball_detected_coordinates) > 0:
                    real_distance_btw_balls = euclidean_dist((ball_detected_coordinates[0], ball_detected_coordinates[1]), (ball_center_coordinate[0], ball_center_coordinate[1]), ((D1+D2)/2)/12)
                    ball_speed = (real_distance_btw_balls/100000)/((time.time()-t)/3600)
                    if ball_speed > speed_limit and time.time()-saved_time > time_limit:
                        human_shot_counter += 1
                        saved_time = time.time()
                        is_human_shot = True
                else:
                    ball_speed = 0
                ball_detected_coordinates = (ball_center_coordinate[0], ball_center_coordinate[1])

                shot_delay_tolerance = 1    # in sec
                flag, coordinate = ball_in_cup(cups_center_coordinates_list=cups_center_coordinate, ball_center_coordinate=ball_center_coordinate, tolerance=35, ball_size=ball_size)
                if flag:
                    cv2.circle(annotated_img, (coordinate[0], coordinate[1]), 50, (0, 255, 0), 4)
                    if is_human_shot and time.time()-saved_time < shot_delay_tolerance:
                        is_human_shot = False
                        human_cups_scored_counter += 1
            else:
                ball_speed = 0
                ball_detected_coordinates = ()

            t = time.time()

            # human number of cups left
            cv2.putText(annotated_img, f'Human: {str(len(human_cups_center_coordinate))}', (110, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # robot number of cups left
            cv2.putText(annotated_img, f'Robot: {str(len(robot_cups_center_coordinate))}', (110, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(annotated_img, f'Human (shot|scored) counter: ({human_shot_counter}|{human_cups_scored_counter})', (110, 640), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            if ball_speed < 0.2:
                ball_speed = 0
            cv2.putText(annotated_img, f'Ball speed (km/h): {abs(round(ball_speed, 2))}', (110, 690), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        else:
            if len(qr_code_corner_coordinate_list) < 2:
                cv2.putText(annotated_img, 'Error: please make free spaces around qr codes', (110, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            elif len(qr_code_corner_coordinate_list) > 2:
                cv2.putText(annotated_img, 'Error: More than 2 qr codes are detected', (110, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # display
        if is_webcam:
            cv2.imshow("annotations", annotated_img)
            cv2.waitKey(1)

        # save
        if saver:
            saver.save_frame(annotated_img)

        iter_end = time.time()
        elapsed_time = 1000 * (iter_end - iter_start)
        # _LOGGER.info(f"Inference {iteration} processed in {elapsed_time} ms")

    if saver:
        saver.close()
    _LOGGER.info(f"Results saved to {save_dir}")


def hough_transform(frame):
    cups_center_point_list = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect cups in the image
    cups = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.8, minDist=22, minRadius=45, maxRadius=55)
    # ensure at least some cups were found
    if cups is not None:
        # convert the (x, y) coordinates and radius of the cups to integers
        cups = np.round(cups[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the cups
        for (x, y, r) in cups:
            # draw the circle in the output image, then draw a small rectangle at its center
            cups_center_point_list.append((x, y))
            # cv2.circle(output, (x, y), r, (0, 0, 255), 4)
            # cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)
    
    return cups_center_point_list


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
    if ball_size < 70:
        if len(ball_center_coordinate) > 0 and len(cups_center_coordinates_list) > 0:
            for c in cups_center_coordinates_list:
                if ball_center_coordinate[0] in range(c[0] - tolerance, c[0] + tolerance) and ball_center_coordinate[1] in range(c[1] - tolerance, c[1] + tolerance):
                    return True, c
    return False, ()


_YOLO_CLASSES = [
    "ball",
    "qr_code"
]


_YOLO_CLASS_COLORS = [(0, 0, 255), (0, 0, 0)]


def main():
    args = parse_args()
    assert len(args.image_shape) == 2
    args.image_shape = tuple(args.image_shape)

    annotate(args)


if __name__ == "__main__":
    main()
