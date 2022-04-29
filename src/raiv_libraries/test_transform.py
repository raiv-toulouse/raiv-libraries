import torchvision
import torchvision.transforms as transforms
import cv2

img = cv2.imread('/common/stockage_image_test/test_success.png')

#Listes des transformations :

transform = transforms.Compose([
            # you can add other transformations in this list
            # transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop(size=224),  # Ancien code, les images font 256x256
            transforms.Resize(size=256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

augmentation = transforms.Compose([
            transforms.CenterCrop(size=224),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
inv_trans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                            transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                                 std=[1., 1., 1.]),
                                            ])

