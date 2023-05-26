import numpy as np
from PIL import Image


class ColorEmbedder:
    """A class for embedding color information from an image."""

    def __init__(self, n_bins: int = 8, alpha: float = 5):
        """
        Initialize the ColorEmbedder object.

        Args:
            n_bins (int):
                Number of color bins to use.
            alpha (float):
                Standard deviation of the noise added to each pixel.

        """
        self.n_bins = n_bins
        self.bins = np.linspace(0, 255, n_bins + 1)
        self.alpha = alpha

    def embed(self, image: Image) -> np.ndarray:
        """
        Embed color information from the given image.

        The embedding process involves resizing the image, converting it to RGB,
        and creating a color histogram. The histogram is composed of n_bins^3
        bins, where n_bins is the number of bins in each dimension (red, green,
        blue). Each pixel in the image is repeated 10 times and some noise is
        added to each pixel. This allows colours near the bin boundaries to fall
        into multiple bins, making the embedding more robust.

        Args:
            image (PIL.Image.Image):
                The input image.

        Returns:
            np.ndarray:
                The flattened color histogram as a 1D numpy array.

        """
        rgb_image = image.convert("RGB").resize(
            (50, 50),
            # resample using nearest neighbour to preserve the original colours.
            # using the default resample method (bicubic) will result in a
            # blurring/blending of colours
            resample=Image.NEAREST,
        )

        pixel_array = np.array(rgb_image).reshape(-1, 3)

        repeated_pixel_array = np.repeat(pixel_array, 10, axis=0)
        noise = np.random.normal(0, self.alpha, repeated_pixel_array.shape)
        pixel_array = repeated_pixel_array + noise

        histogram, _ = np.histogramdd(
            pixel_array,
            bins=[self.bins, self.bins, self.bins],
        )

        # make sure the vector is of unit length
        histogram = histogram / np.linalg.norm(histogram)

        return histogram.flatten()
