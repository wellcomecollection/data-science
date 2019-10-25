# Palette similarity

Uses 5-colour palettes for each image to find most similarly coloured images to
a query image.

## How it works

5-colour palettes are obtained for all images in the collection using a k-means
algorithm on image pixels in
[LAB colour space](https://en.wikipedia.org/wiki/CIELAB_color_space). Euclidean
distances are then calculated between all palette-permutations at query-time.

## Output example

Hitting
[`https://labs.wellcomecollection.org/palette_similarity/by_image_id`](https://labs.wellcomecollection.org/palette_similarity/by_image_id)
will randomly select a query image and return `n=10` nearest neighbours in the
form:

```json
{
  "original_image_id": "L0037645",
  "original_image_url": "https://iiif.wellcomecollection.org/image/L0037645.jpg/full/960,/0/default.jpg",
  "neighbour_ids": [
    "L0037595",
    "L0034492",
    "L0037581",
    "L0063874",
    "L0067338",
    "L0039583",
    "L0069802",
    "L0041575",
    "L0038188",
    "L0082158"
  ],
  "neighbour_urls": [
    "https://iiif.wellcomecollection.org/image/L0037595.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0034492.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0037581.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0063874.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0067338.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0039583.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0069802.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0041575.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0038188.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0082158.jpg/full/960,/0/default.jpg"
  ]
}
```

The API accepts query parameters `image_id`, and `n`.

Alternatively, one can query by palette. Hitting
[`https://labs.wellcomecollection.org/palette_similarity/by_palette`](https://labs.wellcomecollection.org/palette_similarity/by_palette)
will randomly select a hex query palette and return `n=10` nearest neighbours in
the form:

```json
{
  "palette": ["d5a9b6", "7915f8", "31e4f6", "62a05d", "aeb092"],
  "neighbour_ids": [
    "B0007428",
    "L0055060",
    "L0054783",
    "B0009861",
    "L0054124",
    "V0046073ER",
    "W0049936",
    "L0054184",
    "L0031307",
    "L0053208"
  ],
  "neighbour_urls": [
    "https://iiif.wellcomecollection.org/image/B0007428.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0055060.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0054783.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/B0009861.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0054124.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0046073ER.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/W0049936.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0054184.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0031307.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0053208.jpg/full/960,/0/default.jpg"
  ]
}
```

The API accepts query parameters `palette` (a `length=5` list of `length=6` hex
strings comprising a 5-colour palette), and `n` (the number of neighbours to
return).

More detailed API docs can be obtained at
`https://labs.wellcomecollection.org/palette_similarity/docs` or
`https://labs.wellcomecollection.org/palette_similarity/redoc`
