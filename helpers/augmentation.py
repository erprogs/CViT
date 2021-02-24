from albumentations import (
    HorizontalFlip, ImageCompression, VerticalFlip, HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
import numpy as np
from PIL import ImageFile, Image

def strong_aug(p=.5):
    return Compose([
        RandomRotate90(p=0.2),
        Transpose(p=0.2),
        HorizontalFlip(p=0.5),   
        VerticalFlip(p=0.5),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        ShiftScaleRotate(p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.2),
        HueSaturationValue(p=0.2),
    ], p=p)

def augment(aug, image):
    return aug(image=image)['image']

class Aug(object):
    def __call__(self, img):
        aug = strong_aug(p=0.9)
        return Image.fromarray(augment(aug, np.array(img)))