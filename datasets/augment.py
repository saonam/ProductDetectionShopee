import cv2
import numpy as np
import imgaug.augmenters as iaa
import imgaug as ia
import torchvision
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
import random


def random_dilate(img):
    img = np.asarray(img)
    img = cv2.dilate(img, np.ones(
        shape=(random.randint(1, 3), random.randint(1, 3)), dtype=np.uint8))
    return Image.fromarray(img)


def random_erode(img):
    img = np.array(img)
    img = cv2.erode(img, np.ones(shape=(random.randint(
        1, 3), random.randint(1, 3)), dtype=np.uint8))
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
        transformed_img = self.aug.augment_image(img)

        return Image.fromarray(transformed_img)


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude *
                                         random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude *
                                         random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude *
                                         img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude *
                                         img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


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


class Augment(object):
    def __init__(self, width=800, height=800, phase='train'):
        super(Augment, self).__init__()
        self.phase = phase
        self.widht = width
        self.height = height
        self.aug_train = torchvision.transforms.Compose([
            imgaugAugment(),
            transforms.RandomApply(
                [
                    ImageNetPolicy(),
                ],
                p=0.4),

        ])
        self.aug_test = transforms.Compose([
            ResizeImage(height=412, width=412),
            torchvision.transforms.RandomCrop(size=(height, width)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

        ])
        self.testing = transforms.Compose([
            ResizeImageVal(height=height, width=width),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    def transform(self, image):
        if self.phase == 'test':
            image = self.testing(image)
        elif self.phase == 'valid':
            image = self.aug_test(image)
        elif self.phase == 'train':
            image = self.aug_train(image)
            image = self.aug_test(image)
        return image
