import torchvision.transforms as transforms
from PIL import Image
import os as os

i = ''
PATH = '/common/stockage_banque_image/Ventouse_3_souflet/Pression_2_bars/1/fail'  #Write the path folder of resizing
IMAGE_LIST = []
IMAGE_LIST = os.listdir(PATH)

def transform(img):
    transf = transforms.Compose([
        transforms.Resize(size=50),
    ])
    return transf(img)



for i in IMAGE_LIST:
    img = Image.open(PATH + '/' + i)
    image = transform(img)
    image.save(PATH + '/' + i)








