from EfficientNet.model import EfficientNet
from torchvision import transforms as T
import torch
from PIL import Image
from typing import *
import numpy as np
import cv2 

class WeatherClsasifier:
    def __init__(self, weights = "weights/efficientnet/best_weights_256x256_v2.pt", device = "cpu", img_size = (256, 256), labels = ["day", "night", "rain"]):
        if torch.cuda.is_available() and device == "cuda":
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.device = torch.device(self.device)
        self.model = EfficientNet(num_classes = 3, in_channels=3)
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.to(self.device)
        self.model.eval()

        self.labels = labels

        self.img_size = img_size

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ]
        )

    def normalize(self, image: Image.Image):
        return self.transform(image).to(self.device)

    def infer(self, image: Union[Image.Image, np.ndarray]):
        if isinstance(image, np.ndarray):
            image = cv2.resize(image, self.img_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        im = self.normalize(image)
        
        if len(im.shape) == 3:
            im = im[None]

        out = self.model(im)

        _, predicted_labels = torch.max(out.data, dim = 1)

        predicted_labels = predicted_labels.cpu().long().numpy()

        return self.labels[predicted_labels[0]]

if __name__ == "__main__":
    model = WeatherClsasifier(device = "cuda")

    import time 
    img = np.random.randint(0, 256, (2000, 2000, 3)).astype(np.uint8)

    start_time = time.time()

    for i in range(1000):
        # new_img = cv2.resize(img, (256, 256))
        model.infer(img)

    end_time = time.time()

    print((end_time - start_time)/1000)
           