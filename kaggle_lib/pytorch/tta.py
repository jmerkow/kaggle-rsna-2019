from albumentations import HorizontalFlip, DualTransform
from albumentations.augmentations import functional as F


class TTAAlbumentationMixin(object):
    def __call__(self, *args, **kwargs):
        tmp = list(super(TTAAlbumentationMixin, self).__call__(*args, **kwargs).items())
        return {k: [v] for k, v in tmp}


class HorizontalFlipTTA(TTAAlbumentationMixin, HorizontalFlip):

    def __str__(self):
        return "hflip"


class RotateTTA(DualTransform):

    def __str__(self):
        return "{}{}".format('rot', ','.join(map(str, self.angles)))

    def __init__(self, angles, always_apply=False, p=1.0):
        super(RotateTTA, self).__init__(always_apply, p)
        self.angles = angles

    def apply(self, img, **params):
        return [F.rotate(img, angle) for angle in self.angles]

    def get_transform_init_args_names(self):
        return ("angles",)

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError('later')


class FourCropTTA(DualTransform):
    """Crop region from image.
    Args:
        height (int): crop height
        width (int): crop_width
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __str__(self):
        return "5crop"

    def __init__(self, height, width, always_apply=False, p=1.0):
        super(FourCropTTA, self).__init__(always_apply, p)
        self.height = height
        self.width = width

    def get_crops(self, image_shape):
        image_width = image_shape[0]
        image_height = image_shape[1]
        return [dict(x_min=0, y_min=0, x_max=self.width, y_max=self.height),
                dict(x_min=image_width - self.width, y_min=0, x_max=image_width, y_max=self.height),
                dict(x_min=0, y_min=image_height - self.height, x_max=self.width, y_max=image_height),
                dict(x_min=image_width - self.width, y_min=image_height - self.height, x_max=image_width,
                     y_max=image_height),
                # The Fifth Crop:
                # dict(zip(['x_min', 'y_min', 'x_max', 'y_max'],
                # F.get_center_crop_coords(image_height, image_width, self.height, self.width)))
                ]

    def apply(self, img, **params):
        return [F.crop(img, **crop) for crop in self.get_crops(img.shape)]

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError('later')

    #         return F.bbox_crop(bbox, x_min=self.x_min, y_min=self.y_min, x_max=self.x_max, y_max=self.y_max, **params)

    def get_transform_init_args_names(self):
        return ("height", "width")


class ScaleTTA(DualTransform):

    def __str__(self):
        return "{}{}".format('scale', '.'.join(map(str, self.scales)))

    def __init__(self, scales, always_apply=False, p=1.0):
        super(ScaleTTA, self).__init__(always_apply, p)
        self.scales = scales

    def apply(self, img, **params):
        return [F.scale(img, scale) for scale in self.scales]

    def get_transform_init_args_names(self):
        return ("scales",)

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError('later')


class TTATransform(object):

    def __init__(self, hflip=False, five_crop=False, scales=None, angles=None, data_shape=(224, 224)):

        self.transforms = []
        scales = scales or []
        angles = angles or []

        if hflip:
            self.transforms.append(HorizontalFlipTTA())

        if scales:
            self.transforms.append(ScaleTTA(scales))

        if angles:
            self.transforms.append(RotateTTA(angles))

        if five_crop:
            self.transforms.append(FourCropTTA(*data_shape))

    @property
    def name(self):
        if len(self.transforms):
            return '+'.join(str(t) for t in self.transforms)
        return 'none'

    def __str__(self):
        return self.name

    def __call__(self, image):
        images = [image]
        for t in self.transforms:
            images += sum([t(image=i)['image'] for i in images], [])
        return images
