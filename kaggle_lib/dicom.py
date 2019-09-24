import SimpleITK as sitk
import numpy as np
import pydicom
import six

window_presets = {
    # tissueType : [windowWidth, windowCenter]
    'soft_tissue': [350, 50],
    'bone': [1800, 400],
    'brain': [100, 30],
    'post_fossa': [150, 40],
    'temporal_bone': [2800, 600],
    'f2000': [2000, 0],
    'f1000': [1000, 0],
    'min_max': [None, None],
}


def get_pixels_hu(pydcm):
    image = np.stack(pydcm.pixel_array).astype(np.float32)
    # Convert to Hounsfield units (HU)
    intercept = pydcm.RescaleIntercept
    slope = pydcm.RescaleSlope
    if slope != 1:
        image = slope * image
    image += intercept
    return np.array(image, dtype=np.int16)


def pydicom_read_image(filename):
    return get_pixels_hu(pydicom.read_file(filename))


def sitk_read_image(filename):
    return sitk.GetArrayFromImage(sitk.ReadImage(filename)).squeeze().astype('float32')


def window_image(data, width=None, center=None, max_value=255.0, min_value=0.0):
    if width is None:
        width = data.max() - data.min() + 2

    if center is None:
        center = (data.max() + data.min()) / 2

    def _window(x):
        x = ((x - (center - 0.5)) / (width - 1) + 0.5)
        return x * max_value

    return np.piecewise(data,
                        [data <= (center - 0.5 - (width - 1) / 2),
                         data > (center - 0.5 + (width - 1) / 2)],
                        [min_value, max_value, _window])


def windows_as_channels(data, windows, min_value=0., max_value=255.):
    if isinstance(windows, six.string_types):
        windows = [windows]

    def _get_window(w):
        if w is None:
            width = data.max() - data.min()
            return [width, width / 2]
        if isinstance(w, six.string_types):
            assert w in window_presets, 'bad window preset'
            return window_presets[w]
        else:
            assert len(w) == 2, 'bad window values'
            return w

    windows = [_get_window(w) for w in windows]
    return np.dstack([window_image(data, width=width, center=center,
                                   min_value=min_value, max_value=max_value) for width, center in windows])
