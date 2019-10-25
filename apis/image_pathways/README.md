# Image pathways

Uses the feature vectors above to construct visually coherent pathways in feature-space between two images.

## How it works

As a pre-processing step, 4096-dimensional feature vectors are obtained for all images in the collection. These feature vectors are loaded into an [nmslib](https://github.com/nmslib/nmslib) index for fast nearest-neighbour search. [`build_nmslib_index.py`](./build_nmslib_index.py) can be used to construct the index locally if you don't have access to s3.

To find a path, we determine a set of ideal, evenly spaced points in feature space between the start and end image. We then use the nmslib index to rapidly find nearest neighbours to our ideal points, constructing a path from the candidate neighbours which have not yet been included.

## Output example

Hitting [`https://labs.wellcomecollection.org/image_pathways`](https://labs.wellcomecollection.org/image_pathways) will randomly select a start and end image and return a path of length 10 in the form:

```json
{
  "id_path": [
    "V0024122",
    "V0024127",
    "L0025903",
    "V0024316",
    "V0036115",
    "V0035358",
    "V0040723",
    "M0018188",
    "V0016216EL",
    "V0016216ER"
  ],
  "image_url_path": [
    "https://iiif.wellcomecollection.org/image/V0024122.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0024127.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0025903.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0024316.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0036115.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0035358.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0040723.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/M0018188.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0016216EL.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0016216ER.jpg/full/960,/0/default.jpg"
  ]
}
```

The API accepts query parameters `id_1`, `id_2` and `path_length`. Detailed API docs can be obtained at `http://localhost/docs` or `http://localhost/redoc`
