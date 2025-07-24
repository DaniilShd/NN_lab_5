import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
from skimage import io
import numpy as np
import random
import os
from torchvision import transforms

def download_data(path):
    data = []
    for path_image in sorted(os.listdir(path=path)):
        image = Image.open(path + path_image)
        data.append(np.array(image))
    return data

