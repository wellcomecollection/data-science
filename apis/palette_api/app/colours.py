from random import sample
from string import hexdigits

from skimage.color import rgb2lab


def hex_to_rgb(hex_code):
    return [int(hex_code[i: i + 2], 16) for i in range(0, 6, 2)]


def rgb_to_lab(rgb_palette):
    return rgb2lab(rgb_palette.reshape(-1, 1, 3) / 255).squeeze()


def random_hex():
    return ''.join(sample(hexdigits[:16], 6))
