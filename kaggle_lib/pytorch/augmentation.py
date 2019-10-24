import numpy as np
import six
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Compose,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomCrop,
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
    ReplayCompose,
    RandomResizedCrop,
)
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensor

from kaggle_lib.dicom import windows_as_channels


def channel_mean_shift(image):
    if image.ndim == 3:
        return np.dstack([i - i.mean() for i in image.transpose(2, 1, 0)])
    return image - image.mean()


class ChannelMeanShift(ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super(ChannelMeanShift, self).__init__(always_apply, p)

    def apply(self, image, **params):
        return channel_mean_shift(image)

    def get_transform_init_args_names(self):
        return ()


class SerializableReplayCompose(ReplayCompose):

    def __init__(self, *args, **kwargs):
        super(SerializableReplayCompose, self).__init__(*args, **kwargs)

    def _to_dict(self):
        return super(ReplayCompose, self)._to_dict()


def get_preprocessing(preprocessing_fn=None, data_shape=None):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    transform = []

    if data_shape is not None:
        transform.append(PadIfNeeded(min_height=data_shape[1], min_width=data_shape[0]))
        transform.append(CenterCrop(height=data_shape[1], width=data_shape[0]))

    if preprocessing_fn is not None:
        transform.append(Lambda(image=preprocessing_fn), )
    transform.append(ToTensor())

    return Compose(transform)


class ChannelWindowing(ImageOnlyTransform):
    """
    """

    def __init__(
            self, windows=('min_max',),
            force_rgb=True,
            min_pixel_value=0, max_pixel_value=255,
            always_apply=True,
            subtract_mean=False,
            p=1.0
    ):
        super(ChannelWindowing, self).__init__(always_apply, p)

        if isinstance(windows, six.string_types):
            windows = [windows]

        if force_rgb and len(windows) == 1:
            windows = windows * 3

        if force_rgb and len(windows) != 3:
            raise ValueError('Wrong number of channels for force rgb! Must be 1 or 3! got {}'.format(len(windows)))

        self.windows = tuple(windows)
        self.force_rgb = force_rgb
        self.min_pixel_value = min_pixel_value
        self.max_pixel_value = max_pixel_value
        self.subtract_mean = subtract_mean

    def apply(self, image, **params):
        return windows_as_channels(image, self.windows, self.min_pixel_value, self.max_pixel_value,
                                   subtract_mean=self.subtract_mean)

    def get_transform_init_args_names(self):
        return ("windows", "force_rgb", "min_pixel_value", "max_pixel_value", "subtract_mean")


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


# TODO: Change the way all this works....
def make_augmentation(data_shape,
                      resize=None,
                      hflip=0,
                      vflip=0,
                      scale=None,
                      rotate=None,
                      color=None,
                      deform=None,
                      rand_crop=None,
                      rand_resized_crop=None,
                      windows=('soft_tissue',),
                      windows_force_rgb=True,
                      border_mode=4,
                      max_value=1.0,
                      window_max_value=255,
                      channel_mean_shift=False,
                      tta_mode=False,

                      ):
    transforms = []

    if not tta_mode:
        transforms.append(ChannelWindowing(
            windows=windows,
            force_rgb=windows_force_rgb,
            max_pixel_value=window_max_value,
        ))
        if channel_mean_shift:
            transforms.append(ChannelMeanShift())

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
        rotate.setdefault('border_mode', border_mode)
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

    if rand_crop:
        if not isinstance(rand_crop, dict):
            rand_crop = {'p': rand_crop}
        rand_crop.setdefault('p', 1.0)
        r_crop = RandomCrop(height=data_shape[1], width=data_shape[0], **rand_crop)
        transforms.append(r_crop)

    if rand_resized_crop:
        if not isinstance(rand_resized_crop, dict):
            rand_resized_crop = {'p': rand_resized_crop}

        rand_resized_crop.setdefault('scale', (.7, 1.0))
        rand_resized_crop.setdefault('ratio', (3 / 4, 4 / 3))
        rand_resized_crop.setdefault('p', 1.0)
        r_crop = RandomResizedCrop(height=data_shape[1], width=data_shape[0], **rand_crop)
        transforms.append(r_crop)

    c_crop = CenterCrop(height=data_shape[1], width=data_shape[0])
    transforms.append(PadIfNeeded(min_height=data_shape[1], min_width=data_shape[0], border_mode=border_mode))
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

    if not tta_mode:
        transforms.append(ToFloat(max_value=max_value))

    transform = SerializableReplayCompose(transforms)
    return transform


def make_transforms(data_shape, resize=None,
                    windows=('soft_tissue',),
                    windows_force_rgb=True,
                    max_value=1.0,
                    apply_crop=True,
                    apply_windows=True,
                    window_max_value=255,
                    channel_mean_shift=False,
                    border_mode=4,
                    **kwargs):

    transforms = []

    if apply_windows:
        transforms.append(ChannelWindowing(
            windows=windows,
            force_rgb=windows_force_rgb,
            max_pixel_value=window_max_value,
        ))

    if channel_mean_shift:
        transforms.append(ChannelMeanShift())

    if resize == 'auto':
        resize = data_shape

    if resize:
        transforms.append(Resize(*resize))

    if apply_crop:
        transforms.append(CenterCrop(height=data_shape[1], width=data_shape[0]))

    transforms.append(ToFloat(max_value=max_value))

    return Compose(transforms)
