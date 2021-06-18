import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb


def get_5d_coordinates(image):
    rgb_colour_coords = np.array(image).reshape(-1, 3)
    lab_colour_coords = rgb2lab(rgb_colour_coords)
    spatial_coords = [
        [i/image.width, j/image.height]
        for i in range(image.width)
        for j in range(image.height)
    ]
    coords = np.concatenate([lab_colour_coords, spatial_coords], axis=1)
    return coords


def cluster(coords):
    clusterer = KMeans(n_clusters=6).fit(coords)
    return clusterer.cluster_centers_


def get_palette(image):
    coords = get_5d_coordinates(image)
    dominant_points = cluster(coords)
    lab_colour_centres = dominant_points[:, :3]
    colour_centres = (lab2rgb(lab_colour_centres)*255).astype(np.uint8)
    return colour_centres
