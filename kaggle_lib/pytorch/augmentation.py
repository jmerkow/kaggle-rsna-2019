import numpy as np
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Compose,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    RandomBrightness,
    RandomGamma,
    RandomScale,
    RandomContrast,
    Resize,
    Rotate,
    ToFloat,
    PadIfNeeded,
    Lambda,
)
from albumentations.pytorch import ToTensor


def force_rgb(data):
    if data.ndim != 3 and data.ndim != 2:
        raise ValueError("Not a grayscale or RGB image! too many dims! shape: {}".format(str(data.shape)))

    if data.ndim == 3:
        if data.shape[-1] == 3:
            return data
        elif data.shape[-1] == 1:
            data = data[:, :, 0]
        else:
            raise ValueError("Not a grayscale or RGB image! bad dims! shape: {}".format(str(data.shape)))

    data = np.dstack([data, data, data])
    return data


def ForceRGB():
    return Lambda(name='ForceRGB', image=force_rgb)


def make_augmentation(data_shape,
                      resize=None,
                      hflip=0,
                      vflip=0,
                      scale=None,
                      rotate=None,
                      color=None,
                      deform=None,
                      rand_crop=None,
                      max_value=None,
                      min_value=None,
                      ):
    transforms = []

    if resize == 'auto':
        resize = data_shape
    if resize:
        transforms.append(Resize(*resize))

    if hflip:
        transforms.append(HorizontalFlip(p=hflip))

    if vflip:
        transforms.append(VerticalFlip(p=vflip))

    if scale:
        if not isinstance(scale, dict):
            scale = {'scale_limit': scale}
        transforms.append(RandomScale(**scale))

    if rotate:
        if not isinstance(rotate, dict):
            rotate = {'limit': rotate}
        transforms.append(Rotate(**rotate))

    if deform:
        oneof = []
        deform_p = deform.get('p', .3)

        elastic = deform.get('elastic', None)
        grid = deform.get('grid', None)
        optical = deform.get('optical', None)

        if elastic:
            oneof.append(ElasticTransform(**elastic))

        if grid:
            oneof.append(GridDistortion(**grid))

        if optical:
            oneof.append(OpticalDistortion(**optical))

        transforms.append(OneOf(oneof, p=deform_p))

    transforms.append(PadIfNeeded(min_height=data_shape[1], min_width=data_shape[0]))

    c_crop = CenterCrop(height=data_shape[1], width=data_shape[0])
    if rand_crop:
        if not isinstance(rand_crop, dict):
            rand_crop = {'min_max_height': rand_crop}
        transforms.append(OneOf(
            [RandomSizedCrop(height=data_shape[1], width=data_shape[0], **rand_crop),
             c_crop],
            p=1))
    else:
        transforms.append(c_crop)

    if color:
        oneof = []
        color_p = color.get('p', .3)
        contrast = color.get('contrast', None)
        gamma = color.get('gamma', None)
        brightness = color.get('brightness', None)

        if contrast:
            oneof.append(RandomContrast(**contrast))

        if gamma:
            oneof.append(RandomGamma(**gamma))

        if brightness:
            oneof.append(RandomBrightness(**brightness))

        transforms.append(OneOf(oneof, p=color_p))

    if max_value is not None:
        transforms.append(ToFloat(max_value=max_value))

    return Compose(transforms)


def make_transforms(data_shape, resize=None, max_value=None, force_rgb=False, **kwargs):
    transforms = []
    if resize == 'auto':
        resize = data_shape
    if resize:
        transforms.append(Resize(*resize))
    transforms.append(CenterCrop(height=data_shape[1], width=data_shape[0]))

    if max_value is not None:
        transforms.append(ToFloat(max_value=max_value))

    return Compose(transforms)


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    transform = []
    if preprocessing_fn is not None:
        transform.append(Lambda(image=preprocessing_fn), )
    transform.append(ToTensor())

    return Compose(transform)
