

import argparse
import os
import platform
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from TPHYolov5.models.experimental import attempt_load
from TPHYolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from TPHYolov5.utils.general import (LOGGER, apply_classifier, check_file, check_img_size, check_imshow, check_requirements,
                           check_suffix, colorstr, increment_path, non_max_suppression, print_args, save_one_box,
                           scale_coords, strip_optimizer, xyxy2xywh)
from TPHYolov5.utils.plots import colors
from TPHYolov5.utils.torch_utils import load_classifier, select_device, time_sync
from TPHYolov5.utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from typing import Dict, Union


class TPHYolov5:
    def __init__(self, weights: str, device = "cpu", img_size = 1536, half = False):
        self.device = select_device(device)
        self.model = torch.jit.load(weights) if 'torchscript' in weights else attempt_load(weights, map_location=self.device)
        self.half = half
        if half:
            self.model.half()  # to FP16
        self.classes = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(img_size, s=self.stride)  # check image size
        
        if isinstance(self.imgsz, int):
            self.imgsz = (self.imgsz, self.imgsz)

        if self.device.type != "cpu":
            self.model(torch.zeros(1, 3, *self.imgsz).to(device).type_as(next(self.model.parameters())))  # run once
            
    def normalize(self, image: np.ndarray):
        "image: BGR Image from openCv"
        img = letterbox(image, self.imgsz, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255
        
        return img
    
    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv.rectangle(image, p1, p2, color, thickness=lw, lineType=cv.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv.rectangle(image, p1, p2, color, -1, cv.LINE_AA)  # filled
            cv.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                        thickness=tf, lineType=cv.LINE_AA)
    
    def visualize(self, original_image: np.ndarray, det, hide_labels = False, hide_conf = False):
        image = original_image.copy()
        for *xyxy, conf, cls in reversed(det):
            thickness = max(1, min(image.shape[:2])//500)
            
            class_num = int(cls)
            
            label = None if hide_labels else (self.classes[class_num] if hide_conf else f'{self.classes[class_num]} {conf:.2f}')
            
            self.plot_box_and_label(image, thickness, xyxy, label, color = colors(class_num, True))
        
        return image
    
    def infer(self, batch_image: np.ndarray, conf_thresh = 0.25, iou_thresh = 0.45, augment = False, visualize = False):
        "image: BGR Image from openCv"
        
        if len(batch_image.shape) == 3:
            batch_image = batch_image[None]

        batch_normalized_image = []

        for image in batch_image:
            batch_normalized_image.append(
                self.normalize(image)
            )
        
        batch_normalized_image = torch.stack(batch_normalized_image, dim = 0)

        pred = self.model(batch_normalized_image, augment=augment, visualize=visualize)[0]

        pred = pred.reshape(1, -1, 11)

        pred = non_max_suppression(pred, conf_thresh, iou_thresh, max_det = 1000)
        
        results = {
                "FPS": 0,
                "status": "day",
                "person": 0,
                "bicycle": 0,
                "car": 0,
                "motorcycle": 0,
                "bus": 0,
                "truck": 0
            }
        
        if len(pred):
            
            det = pred[0].cpu()
            
            det[:, :4] = scale_coords(batch_normalized_image.shape[2:], det[:, :4], image.shape).round()
            
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

                if self.classes[int(c)] in results.keys():
                    results[self.classes[int(c)]] = n.cpu().numpy()

        return results, det
    
if __name__ == '__main__':
    device = 0 if torch.cuda.is_available() else 'cpu'
    import os
    model = TPHYolov5(weights = "weights/yolo/best.pt", device = device)
    for fname in os.listdir("images"):
        print(f"------------{fname}----------------")
        image = cv.imread(f"images//{fname}")
        results, det = model.infer(image, conf_thresh = 0.25)
        new_image = model.visualize(image, det, hide_labels = True, hide_conf = True)
        cv.imwrite(f"results//images//{fname}", new_image)






 