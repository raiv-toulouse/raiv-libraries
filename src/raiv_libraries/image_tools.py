import torchvision.transforms as transforms
from torchvision.transforms.functional import crop


class ImageTools:
    CROP_WIDTH = 50  # Width and height for rgb and depth cropped images
    CROP_HEIGHT = 50
    IMAGE_SIZE_FOR_NN = 224
    IMAGE_SIZE_BEFORE_CROP = 256

    # Transforms used to process images before training or inference
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.Resize(size=IMAGE_SIZE_BEFORE_CROP),
        transforms.CenterCrop(size=IMAGE_SIZE_FOR_NN),  # Ancien code, les images font 254*254
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    augmentation = transforms.Compose([
        transforms.Resize(size=IMAGE_SIZE_BEFORE_CROP),
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop(size=IMAGE_SIZE_FOR_NN),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_image = transforms.Compose([
        transforms.Resize(size=IMAGE_SIZE_BEFORE_CROP),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # Used to correctly display images
    inv_trans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                              std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                    transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                              std=[1., 1., 1.]),
                                    ])

    @staticmethod
    def crop_xy(image, x, y):
        """ Crop image at position (predict_center_x,predict_center_y) and with size (WIDTH,HEIGHT) """
        return crop(image, y - ImageTools.CROP_HEIGHT/2, x - ImageTools.CROP_WIDTH/2,
                    ImageTools.CROP_HEIGHT, ImageTools.CROP_WIDTH)  # top, left, height, width