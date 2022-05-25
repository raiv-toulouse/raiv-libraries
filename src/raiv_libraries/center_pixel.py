import numpy as np
from PIL import Image
import os as os


i = ''
PATH = '/common/Random_images_picks/check/success'  #Write the path folder of resizing
IMAGE_LIST = []
IMAGE_LIST = os.listdir(PATH)

def pixel_change(image_data):
    pixel = [0, 204, 0]
    image_data[24][25] = pixel
    image_data[24][24] = pixel
    image_data[25][24] = pixel
    image_data[25][25] = pixel
    return image_data

for i in IMAGE_LIST:
    img = Image.open(PATH + '/' + i)
    image_data = np.asarray(img)
    image_changed = pixel_change(image_data)
    image = Image.fromarray(image_changed)
    image.save(PATH + '/' + i)
