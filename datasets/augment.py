import cv2
import numpy as np
import imgaug.augmenters as iaa
import imgaug as ia
import torchvision
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
from RandAugment.augmentations import Lighting, RandAugment



class ResizeImage(object):
    def __init__(self, height=256, width=256):
        self.height = height
        self.width = width

    def __call__(self, img):
        img = np.array(img)
        h, w = img.shape[:2]
        if h < w:
            w = int(self.height*w*1.0/h)
            h = self.height
        else:
            h = int(self.width*h*1.0/w)
            w = self.width

        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

        return Image.fromarray(img)


class ResizeImageVal(object):
    def __init__(self, height=256, width=256):
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


def sometimes(aug): return iaa.Sometimes(0.5, aug)


class imgaugAugment(object):
    def __init__(self):
        super(imgaugAugment, self).__init__()
        self.seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.2),  # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # translate by -20 to +20 percent (per axis)
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=(-16, 16),  # shear by -16 to +16 degrees
                    # use nearest neighbour or bilinear interpolation (fast)
                    order=[0, 1],
                    # if mode is constant, use a cval between 0 and 255
                    cval=(0, 255),
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    mode=ia.ALL
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                    # convert images into their superpixel representation
                    sometimes(iaa.Superpixels(
                        p_replace=(0, 1.0), n_segments=(20, 200))),
                    iaa.OneOf([
                        # blur images with a sigma between 0 and 3.0
                        iaa.GaussianBlur((0, 3.0)),
                        # blur image using local means with kernel sizes between 2 and 7
                        iaa.AverageBlur(k=(2, 7)),
                        # blur image using local medians with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)),
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(
                        0.75, 1.5)),  # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(
                        0, 2.0)),  # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(
                            alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    # add gaussian noise to images
                    iaa.AdditiveGaussianNoise(loc=0, scale=(
                        0.0, 0.05*255), per_channel=0.5),
                    iaa.OneOf([
                        # randomly remove up to 10% of the pixels
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(
                            0.02, 0.05), per_channel=0.2),
                    ]),
                    # invert color channels
                    iaa.Invert(0.05, per_channel=True),
                    # change brightness of images (by -10 to 10 of original value)
                    iaa.Add((-10, 10), per_channel=0.5),
                    # change hue and saturation
                    iaa.AddToHueAndSaturation((-20, 20)),
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.LinearContrast((0.5, 2.0))
                        )
                    ]),
                    # improve or worsen the contrast
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    # move pixels locally around (with random strengths)
                    sometimes(iaa.ElasticTransformation(
                        alpha=(0.5, 3.5), sigma=0.25)),
                    # sometimes move parts of the image around
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                    random_order=True
                )
            ],
            random_order=True
        )

    def __call__(self, img):
        img = self.seq.augment_image(img)
        return Image.fromarray(img)

_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
class Augment(object):
    def __init__(self, width=320, height=320, phase='train'):
        super(Augment, self).__init__()
        self.phase = phase
        self.widht = width
        self.height = height
        # self.transform_train = torchvision.transforms.Compose([
        #     imgaugAugment(),


        # ])
        self.transform_train = transforms.Compose([
            RandAugment(n=3, m=9),
            transforms.RandomResizedCrop(self.height, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),

        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(self.height+32, interpolation=Image.BICUBIC),
            transforms.CenterCrop(self.height),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    def __call__(self, image):
        if self.phase == 'train':
            image = self.transform_train(image)
        elif self.phase == 'valid' or self.phase=='test':
            image = self.transform_test(image)
        return image
