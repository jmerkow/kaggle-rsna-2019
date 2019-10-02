def convert_to_hu(img, slope, intercept, min_val=-2000):
    if min_val is not None:
        img[img == min_val] = 0
    if slope != 1:
        img *= slope
    img += intercept
    return img


def window_image(img, window_center, window_width, intercept=0, slope=1, min_val=-2000):
    img = convert_to_hu(img, slope, intercept, min_val=min_val)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img
