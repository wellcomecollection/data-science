import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


class ColorEmbedder:
    def __init__(self, n_bins=8, gaussian_sigma=3):
        self.n_bins = n_bins
        self.bins = np.linspace(0, 255, n_bins + 1)
        self.gaussian_sigma = gaussian_sigma

    def embed(self, image: Image) -> np.ndarray:
        """
        Embed an image as a vector of colour frequencies.
        :param image: a PIL image
        :return: a vector of length 8
        """
        rgb_image = image.convert("RGB").resize(
            (50, 50),
            # resample using nearest neighbour to preserve the original colours.
            # using the default resample method (bicubic) will result in a
            # blending of colours
            resample=Image.NEAREST,
        )

        pixel_array = np.array(rgb_image).reshape(-1, 3)
        histogram, _ = np.histogramdd(
            pixel_array, 
            bins=[self.bins, self.bins, self.bins],
        )

        # run a very weak gaussian filter over the histogram to allow some
        # tolerance for similar colours
        histogram = gaussian_filter(histogram, sigma=self.gaussian_sigma)

        # flatten the histogram
        return histogram.flatten()
