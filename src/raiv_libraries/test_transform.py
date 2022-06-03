import torchvision
import torchvision.transforms as transforms
from PIL import Image

img = Image.open('/common/stockage_image_test/theo.png')

def transform(img):
    transf = transforms.Compose([
        # you can add other transformations in this list
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),  # Ancien code, les images font 256x256
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transf(img)

image = transform(img)  # Get the loaded images, resize in 256 and transformed in tensor
pil_image = torchvision.transforms.ToPILImage()(image)
pil_image.save("/common/stockage_image_test/theo2.png", format="png")






#inv_trans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
#                                           transforms.Normalize(mean=[-0.485, -0.456, -0.406],
#                                                                 std=[1., 1., 1.]),
#                                            ])

