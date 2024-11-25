import argparse
import os
import time

import cv2 as cv
import numpy as np
import torch

from classifier import WeatherClsasifier
from detector import TPHYolov5
from enlighten import EnlightenModel


def display_results(img, results):

    ratio = max(img.shape[:2]) / 960

    expanded_area = np.ones((img.shape[0], int(150 * ratio), 3), dtype=np.uint8) * 255

    new_img = np.concatenate([img, expanded_area], axis=1)

    line_spacing = 20 * ratio
    thickness = int(ratio)
    line = 1

    for key, value in results.items():
        if key == "FPS":
            continue
        cv.putText(
            new_img,
            f"{key}: {value}",
            (img.shape[1] + int(10 * ratio), int(line_spacing * line)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5 * ratio,
            (0, 0, 255),
            thickness=thickness,
        )
        line += 1

    return new_img


def print_results(results, time=None):
    s = ""
    for key, value in results.items():
        s += f"{key}: {value}, "

    if time is not None:
        s += f"time: {time * 1000:.2f}ms"
    else:
        s = s.strip().strip(",")

    print(s)


def detect_night_image(
    detector: TPHYolov5,
    EnlightenModel: EnlightenModel,
    night_image: np.ndarray,
    conf_thresh=0.25,
    iou_thresh=0.45,
):
    "night_image: bgr image from openCV"
    image_enhancement = EnlightenModel.infer(night_image)

    batch_image = np.stack([night_image, image_enhancement], axis=0)

    results, det = detector.infer(
        batch_image, conf_thresh=conf_thresh, iou_thresh=iou_thresh
    )

    return results, det


def detect_video(
    source,
    weights,
    scale_factor=1.0,
    imgsz=1996,  # inference size (pixels)
    device="cpu",
    conf_thresh=0.25,  # confidence threshold
    iou_thresh=0.45,  # NMS IOU threshold
    savepath=None,  # path to save images/videos
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    vid_stride=1,  #
    histogram_thresh=25,
    night_histogram_ratio=0.5,
    **kwargs,
):
    if device != "cpu":
        device = device if torch.cuda.is_available() else "cpu"
    tph_yolov5_model = TPHYolov5(
        weights=weights, img_size=imgsz, device="cuda:0" if device == "cuda" else device, half=half
    )

    enlighten_model = EnlightenModel(device)

    weather_classifier = WeatherClsasifier(device)

    video = cv.VideoCapture(source)

    fps = video.get(cv.CAP_PROP_FPS)
    w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    scaled_w = int(scale_factor * w)
    scaled_h = int(scale_factor * h)

    if savepath is not None:
        dummy_image = np.zeros((scaled_h, scaled_w, 3))

        new_dummy_image = display_results(dummy_image, results={})

        h, w = new_dummy_image.shape[:2]
        vid_writer = cv.VideoWriter(
            savepath, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

    frame_counter = 0

    pre_results = None
    pre_det = None

    processing_time = 0

    while True:
        ret, frame = video.read()

        if ret:
            frame = cv.resize(frame, (scaled_w, scaled_h))

            if frame_counter % vid_stride == 0:
                start_time = time.time()

                status = weather_classifier.infer(frame)

                if status == "night":

                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                    histogram = np.histogram(gray, bins=256)[0]

                    night_ratio = np.sum(
                        histogram[:histogram_thresh], dtype=np.float32
                    ) / np.sum(histogram, dtype=np.float32)

                    print("histogram ratio", night_ratio, end=" ")

                    if night_ratio >= night_histogram_ratio:
                        results, det = detect_night_image(
                            tph_yolov5_model,
                            enlighten_model,
                            frame,
                            conf_thresh=conf_thresh,
                            iou_thresh=iou_thresh,
                        )

                    else:
                        results, det = tph_yolov5_model.infer(
                            frame, conf_thresh=conf_thresh, iou_thresh=iou_thresh
                        )

                else:
                    results, det = tph_yolov5_model.infer(
                        frame, conf_thresh=conf_thresh, iou_thresh=iou_thresh
                    )

                visualized_image = tph_yolov5_model.visualize(
                    frame, det, hide_labels=hide_labels, hide_conf=hide_conf
                )

                end_time = time.time()

                results["status"] = status
                results["FPS"] = round(1 / (end_time - start_time) * vid_stride, 2)

                processing_time = end_time - start_time

                save_image = display_results(visualized_image, results)

                pre_det = det
                pre_results = results
            else:
                visualized_image = tph_yolov5_model.visualize(
                    frame, pre_det, hide_labels=hide_labels, hide_conf=hide_conf
                )
                save_image = display_results(visualized_image, pre_results)

            print_results(pre_results, time=processing_time)
            frame_counter += 1

            if savepath is not None:
                vid_writer.write(save_image)
        else:
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Detect objects in a video.")

    parser.add_argument(
        "--source", type=str, required=True, help="path to input video file"
    )
    parser.add_argument(
        "--weights", type=str, default="weights/yolo/best.pt", help="path to YOLO weights file"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1536, help="inference size, [1280, 1536, 1996]"
    )
    parser.add_argument(
        "--scale_factor", type=float, default=1.0, help="factor to scale video size"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="device to run on (cuda or cpu)"
    )
    parser.add_argument(
        "--conf_thresh", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou_thresh", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--savedir", type=str, default=None, help="direction to save results"
    )
    parser.add_argument("--hide_labels", action="store_true", help="hide labels")
    parser.add_argument("--hide_conf", action="store_true", help="hide confidences")
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument(
        "--vid_stride", type=int, default=1, help="video frame-rate stride"
    )
    parser.add_argument(
        "--histogram_thresh", type=int, default=25, help="histogram threshold"
    )
    parser.add_argument(
        "--night_histogram_ratio", type=float, default=0.6, help="night histogram ratio"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    if args.savedir is not None:
        if not os.path.exists(args.savedir):
            os.mkdir(args.savedir)
        filename = os.path.basename(args.source)  # Lấy tên tệp từ source
        args.savepath = os.path.join(args.savedir, filename)  # Kết hợp savedir và filename

    kwargs = vars(args)  # vars() chuyển đổi object thành dictionary

    print(kwargs)

    # detect_video(**kwargs)
