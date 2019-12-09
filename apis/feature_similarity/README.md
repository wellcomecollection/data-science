# Image similarity

Uses VGG16 feature vectors to find visually similar images to a query image.

## How it works

4096-dimensional feature vectors are obtained for all images in the collection using a pretrained VGG16 pytorch model, before being indexed into an [nmslib](https://github.com/nmslib/nmslib) object, allowing for fast nearest-neighbour matching.

## Output example

Hitting [`https://labs.wellcomecollection.org/image_similarity`](https://labs.wellcomecollection.org/image_similarity) will randomly select a query image and return `n=10` nearest neighbours in the form:

```json
{
  "original_image_id": "V0050533",
  "original_image_url": "https://iiif.wellcomecollection.org/image/V0050533.jpg/full/960,/0/default.jpg",
  "neighbour_ids": [
    "V0025239",
    "V0019277",
    "V0050414",
    "V0024693",
    "L0035996",
    "V0023321",
    "V0007628ETL",
    "V0041020",
    "V0019275",
    "V0025134"
  ],
  "neighbour_urls": [
    "https://iiif.wellcomecollection.org/image/V0025239.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0019277.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0050414.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0024693.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0035996.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0023321.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0007628ETL.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0041020.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0019275.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0025134.jpg/full/960,/0/default.jpg"
  ]
}
```