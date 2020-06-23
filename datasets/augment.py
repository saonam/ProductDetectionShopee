import imgaug.augmenters as iaa
import imgaug as ia
import torchvision as tv

class Augment(object):
    def __init__(self, phase='train', width=512, height=512):
        super(Augment, self).__init__()
        self.phase = phase
        self.widht = width
        self.height = height
        self.aug = iaa.Sequential([
                # flip
                iaa.Fliplr(0.5),
                iaa.Flipud(0.2),
                iaa.OneOf([
                    iaa.Dropout([0.05, 0.2]), # drop 5% or 20% of all pixels
                    iaa.Sharpen((0.0, 1.0)),       # sharpen the image
                ]),


                # blur
                iaa.SomeOf(2, [
                    iaa.GaussianBlur(sigma=(0.0, 3.0)),
#         iaa.AverageBlur(k=((5, 11), (1, 3))),
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                # collections randaugment
#     iaa.RandAugment(m=(0, 9))
            iaa.Resize({"height": 224, "width": 224})
            ])

    def transform(self, image):
        if self.phase=='train':
            image = self.aug(image=image)
        image = tv.transforms.ToTensor()(image)

        return image
