import time
import os
from EnlightenGAN.options.test_options import TestOptions
from EnlightenGAN.data.data_loader import CreateDataLoader
from EnlightenGAN.models.models import create_model
from EnlightenGAN.util.visualizer import Visualizer
from pdb import set_trace as st
from EnlightenGAN.util import html
import torch
import numpy as np
import random
from EnlightenGAN.data.base_dataset import BaseDataset, get_transform
import cv2 as cv
    
class EnlightenModel(object):
    def __init__(self, device = 'cpu'):
        
        if torch.cuda.is_available() and device == "cuda":
            device = 0
        elif not torch.cuda.is_available() or device == "cpu":
            device = -1
        
        self.device = device
        
        self.opt = self.create_opt()
        
        self.get_transform = get_transform(self.opt)
        
        self.model = create_model(self.opt)

    def create_opt(self):
        
        arguments = ["--dataroot", "",'--name', 'enlightening', '--model', 'single', "--which_direction", "AtoB", "--no_dropout", 
                    "--dataset_mode", "unaligned", "--which_model_netG", "sid_unet_resize", "--skip", "1",
                    "--use_norm", "1", "--use_wgan", "0", "--self_attention", "--times_residual", "--instance_norm", "0",
                    "--resize_or_crop", "no", "--which_epoch", "200", "--gpu_ids", f"{self.device}"]
        
        opt = TestOptions().parse(arguments)
        
        opt.nThreads = 1   # test code only supports nThreads = 1
        opt.batchSize = 1  # test code only supports batchSize = 1
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip
        
        return opt

    def normalize_image(self, rgb_image):
        A_img = self.get_transform(rgb_image)
        
        if self.opt.resize_or_crop == 'no':
                r,g,b = A_img[0]+1, A_img[1]+1, A_img[2]+1
                A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
                A_gray = torch.unsqueeze(A_gray, 0)
                input_img = A_img
        else:
            w = A_img.size(2)
            h = A_img.size(1)
            
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
            if self.opt.vary == 1 and (not self.opt.no_flip) and random.random() < 0.5:
                times = random.randint(self.opt.low_times,self.opt.high_times)/100.
                input_img = (A_img+1)/2./times
                input_img = input_img*2-1
            else:
                input_img = A_img
            
            r,g,b = input_img[0]+1, input_img[1]+1, input_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)

        A_gray = A_gray[None, ...]
        A_img = A_img[None, ...]
        input_img = input_img[None, ...]
        B_img = A_img.clone()
        return {'A': A_img, 'B': B_img, 'A_gray': A_gray, 'input_img': input_img,
                'A_paths': "", 'B_paths': ""}
        
    def infer(self, bgr_image, size = None):
                
        rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
        
        if size is not None:
            rgb_image = cv.resize(rgb_image, size)
        
        data = self.normalize_image(rgb_image)
        
        self.model.set_input(data)
        
        out = self.model.predict()
        
        image_enhancement = out["fake_B"]
        
        image_enhancement = cv.cvtColor(image_enhancement, cv.COLOR_RGB2BGR)
        
        return image_enhancement

if __name__ == "__main__":  
    model = EnlightenModel(device = "cuda")
    for fname in ["nepal_real_A.png", "pic2_real_A.png"]:
        img = cv.imread(f"images/{fname}")
        
        out = model.infer(img)

        cv.imwrite(f"results/enlighten/{fname}", out)
    