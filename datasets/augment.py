import cv2
import numpy as np
import imgaug.augmenters as iaa
import imgaug as ia
import torchvision
from torchvision import transforms
from PIL import Image
def random_dilate(img):
    img = np.asarray(img)
    img = cv2.dilate(img, np.ones(shape=(random.randint(1,3), random.randint(1,3)), dtype=np.uint8))
    return Image.fromarray(img)

def random_erode(img):
    img = np.array(img)
    img = cv2.erode(img, np.ones(shape=(random.randint(1,3), random.randint(1,3)), dtype=np.uint8))
    return Image.fromarray(img)

class ImgAugBlur(object):
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.5,
                          iaa.OneOf(
                              [
                                  iaa.GaussianBlur(sigma=(0, 3.0)),
                                  iaa.AverageBlur(k=(3, 11)),
                                  iaa.MedianBlur(k=(3, 11))
                              ])
                          ),

        ])

    def __call__(self, img):
        img = np.array(img)
        transformed_img =  self.aug.augment_image(img)

        return Image.fromarray(transformed_img)

class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.
        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
    
class ResizeImage(object):
    def __init__(self, height=512, width=512):
        self.height = height 
        self.width = width
        self.pad_fix = iaa.PadToFixedSize(width=width, height=height)
    def __call__(self, img):
        img = np.array(img)
        h, w = img.shape[:2]
        if h > w:
            w = int(self.height*w*1.0/h)
            h = self.height
        else:
            h = int(self.width*h*1.0/w)
            w = self.width
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        img = self.pad_fix.augment_image(img)

        return Image.fromarray(img)

class Augment(object):
    def __init__(self, width=512, height=512, phase='train'):
        super(Augment, self).__init__()
        self.phase = phase
        self.widht = width
        self.height = height
        self.aug_train = torchvision.transforms.Compose([
            transforms.RandomApply(
                        [
                            random_dilate,
                        ],
                        p=0.15),

                    transforms.RandomApply(
                        [
                            random_erode,
                        ],
                        p=0.15),
            transforms.RandomApply(
                        [
                            ImgAugBlur(),
                        ],
                        p=0.4),
            transforms.RandomApply(
                        [
                            ImageNetPolicy(),
                        ],
                        p=0.4),
            transforms.RandomAffine(degrees=5, scale=(0.9, 1.1), shear=5, resample=Image.NEAREST, fillcolor=255),
            ResizeImage(width=width, height=height),
            
        ])
        self.aug_test = transforms.Compose([
            ResizeImage(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.ToTensor(),
            
        ])

    def transform(self, image):
        if self.phase=='train':
            image = self.aug_train(image)
        image = self.aug_test(image)
        return image
