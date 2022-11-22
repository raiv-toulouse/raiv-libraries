import torchvision.transforms as transforms
from torchvision.transforms.functional import crop
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import torchvision.transforms.functional as F
import torch


class ImageTools:
    CROP_WIDTH = 50  # Width and height for rgb and depth cropped images
    CROP_HEIGHT = 50
    IMAGE_SIZE_FOR_NN = 224
    IMAGE_SIZE_BEFORE_CROP = 256
    INITIAL_WIDTH = 640
    INITIAL_HEIGHT = 480

    tranform_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Transforms used to process images before training or inference
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.Resize(size=IMAGE_SIZE_BEFORE_CROP),
        transforms.CenterCrop(size=IMAGE_SIZE_FOR_NN),  # Ancien code, les images font 254*254
        transforms.ToTensor(),
        tranform_normalize
    ])

    augmentation = transforms.Compose([
        transforms.Resize(size=IMAGE_SIZE_BEFORE_CROP),
        #transforms.RandomRotation(degrees=15),
        transforms.RandomCrop(size=IMAGE_SIZE_FOR_NN),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        tranform_normalize
    ])

    transform_image = transforms.Compose([
        transforms.Resize(size=IMAGE_SIZE_FOR_NN),
        transforms.ToTensor(),
        tranform_normalize
    ])


    # Used to correctly display images
    inv_trans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                              std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                    transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                              std=[1., 1., 1.]),
                                    ])

    @staticmethod
    def crop_xy(image, x, y, crop_width, crop_height):
        """ Crop image PIL at position (predict_center_x,predict_center_y) and with size (WIDTH,HEIGHT) """
        return crop(image, y - crop_height/2, x - crop_width/2, crop_height, crop_width)  # top, left, height, width

    @staticmethod
    def center_crop(image, crop_width, crop_height):
        return ImageTools.crop_xy(image, image.width // 2, image.height // 2, crop_width, crop_height)

    @staticmethod
    def pil_to_opencv(pil_image):
        #return cv2.cvtColor(np.asarray(pil_image),cv2.COLOR_RGB2BGR)
        return np.asarray(pil_image)

    @staticmethod
    def ros_msg_to_pil(msg):
        """ Recover the image in the msg sensor_msgs.Image message and convert it to a PILImage"""
        size = (msg.width, msg.height)  # Image size
        img = Image.frombytes('RGB', size, msg.data)  # sensor_msg Image to PILImage
        return img

    @staticmethod
    def numpy_to_pil(numpy):
        return Image.fromarray(numpy)

    @staticmethod
    def pil_to_numpy(pil_image):
        return np.array(pil_image)

    @staticmethod
    def opencv_to_pil(opencv_image):
        return Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))

    @staticmethod
    def tensor_to_pil(tensor_image):
        return F.to_pil_image(tensor_image)

    @staticmethod
    def show_image(imgs, files=None, title='Images', inv_needed=True):
        """
        Display image(s) in a matplotlib window.
        If files, display the original files (directly read from disk)
        Image can be of type : opencv, PIL, list of images, tensor [3,W,H], tensor [batch_size, 3, W, H]
        For tensor, if inv_needed is True, appli the denormalization transform
        """
        matplotlib.use('Qt5Agg')
        if not isinstance(imgs, list) and not (isinstance(imgs, torch.Tensor) and imgs.ndimension() == 4) : # not a LIST or not a Tensor type : [batch_size, nb_channels, width, height]
            imgs = [imgs]
        fix, axs = plt.subplots(nrows=2 if files else 1, ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            if isinstance(img, torch.Tensor):  # Tensor Image
                img = img.detach()
                if inv_needed:
                    img = ImageTools.inv_trans(img)
                img = F.to_pil_image(img)
            elif isinstance(img, np.ndarray):  # OpenCV image
                img = ImageTools.opencv_to_pil(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            if files:
                img_file = Image.open(files[i])
                axs[1, i].imshow(np.asarray(img_file))
                axs[1, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.suptitle(title)
        plt.show()

###################################################################################################################
# Main program
###################################################################################################################

if __name__ == '__main__':
    import argparse
    from torchvision.io import read_image

    parser = argparse.ArgumentParser(description='Test different image conversions.')
    parser.add_argument('img_file', type=str, help='Image file')
    args = parser.parse_args()

    # OPENCV -> PIL
    img_opencv = cv2.imread(args.img_file)
    img_pil = ImageTools.opencv_to_pil(img_opencv)
    #img_pil.show('Image PIL')

    img_tensor = read_image(args.img_file)
    ImageTools.show_image(img_tensor, 'Tensor')

    # PIL -> OpenCV
    img_pil = Image.open(args.img_file)
    ImageTools.show_image(img_pil, 'PIL')
    img_opencv = ImageTools.pil_to_opencv(img_pil)
    ImageTools.show_image(img_opencv, 'OpenCV')

    ImageTools.show_image([img_pil, img_tensor, img_opencv], 'All')
    # Display an OPenCV window
    cv2.imshow("OpenCV",img_opencv)
    cv2.waitKey()

