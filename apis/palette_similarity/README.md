# Palette similarity

Uses 5-colour palettes for each image to find most similarly coloured images to
a query image.

## How it works

5-colour palettes are obtained for all images in the collection using a k-means algorithm on image pixels in [LAB colour space](https://en.wikipedia.org/wiki/CIELAB_color_space). 5x3 dimensional LAB palettes are then embedded into a single dimensional palette embedding (see research notebooks), and stored in an nmslib index. A nearest neighbour approximation of euclidean distance between all palette embeddings is then calculated at query-time.

## Output example

Hitting
[`https://labs.wellcomecollection.org/palette-similarity/works/{catalogue_id}`](https://labs.wellcomecollection.org/palette-similarity/works/{catalogue_id})
will return the `n=10` nearest neighbours to the given image (`catalogue_id`) in the
form:

```json
{
  "original": {
    "miro_id": "L0030365",
    "catalogue_id": "dgxthrkq",
    "miro_uri": "https://iiif.wellcomecollection.org/image/L0030365.jpg/full/960,/0/default.jpg",
    "catalogue_uri": "https://wellcomecollection.org/works/dgxthrkq"
  },
  "neighbours": [
    {
      "miro_id": "L0030814",
      "catalogue_id": "mw8hxa8g",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0030814.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/mw8hxa8g"
    },
    {
      "miro_id": "V0033889EL",
      "catalogue_id": "b25hfd5k",
      "miro_uri": "https://iiif.wellcomecollection.org/image/V0033889EL.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/b25hfd5k"
    },
    {
      "miro_id": "L0015306",
      "catalogue_id": "b2ur2ebp",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0015306.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/b2ur2ebp"
    },
    {
      "miro_id": "L0052696",
      "catalogue_id": "vmvbxwp2",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0052696.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/vmvbxwp2"
    },
    {
      "miro_id": "L0063538",
      "catalogue_id": "dkuwfy63",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0063538.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/dkuwfy63"
    },
    {
      "miro_id": "V0025632",
      "catalogue_id": "pcyc5hbe",
      "miro_uri": "https://iiif.wellcomecollection.org/image/V0025632.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/pcyc5hbe"
    },
    {
      "miro_id": "L0063535",
      "catalogue_id": "vxdsp8ck",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0063535.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/vxdsp8ck"
    },
    {
      "miro_id": "L0082412",
      "catalogue_id": "nbnbxptp",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0082412.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/nbnbxptp"
    },
    {
      "miro_id": "L0053992",
      "catalogue_id": "qv2v376h",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0053992.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/qv2v376h"
    },
    {
      "miro_id": "L0030803",
      "catalogue_id": "ruaq3gnw",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0030803.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/ruaq3gnw"
    }
  ]
}
```

The similarity accepts query parameters `catalogue_id`, and `n`.

Alternatively, one can query by palette. Hitting
[`https://labs.wellcomecollection.org/palette-similarity/palette`](https://labs.wellcomecollection.org/palette_similarity/by_palette)
will randomly select a hex query palette and return `n=10` nearest neighbours in
the form:

```json
{
  "original": {
    "palette": ["f7c490", "8b306a", "9782a3", "0924d1", "b3c9a7"]
  },
  "neighbours": [
    {
      "miro_id": "B0009765",
      "catalogue_id": "ycc9gkw3",
      "miro_uri": "https://iiif.wellcomecollection.org/image/B0009765.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/ycc9gkw3"
    },
    {
      "miro_id": "L0037076",
      "catalogue_id": "nkwg6t7f",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0037076.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/nkwg6t7f"
    },
    {
      "miro_id": "L0058991",
      "catalogue_id": "h8cn6zes",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0058991.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/h8cn6zes"
    },
    {
      "miro_id": "L0030349",
      "catalogue_id": "gcz9g9pk",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0030349.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/gcz9g9pk"
    },
    {
      "miro_id": "L0065278",
      "catalogue_id": "n9mrfmp4",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0065278.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/n9mrfmp4"
    },
    {
      "miro_id": "A0000409",
      "catalogue_id": "qqym8sb3",
      "miro_uri": "https://iiif.wellcomecollection.org/image/A0000409.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/qqym8sb3"
    },
    {
      "miro_id": "L0059071",
      "catalogue_id": "tzs35z37",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0059071.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/tzs35z37"
    },
    {
      "miro_id": "L0058992",
      "catalogue_id": "z95j5tb4",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0058992.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/z95j5tb4"
    },
    {
      "miro_id": "B0004798",
      "catalogue_id": "fqjs8eft",
      "miro_uri": "https://iiif.wellcomecollection.org/image/B0004798.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/fqjs8eft"
    },
    {
      "miro_id": "L0059062",
      "catalogue_id": "zrvr9ry8",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0059062.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/zrvr9ry8"
    }
  ]
}
```

The similarity accepts query parameters `palette` (a `length=5` list of `length=6` hex
strings comprising a 5-colour palette), and `n` (the number of neighbours to
return).

More detailed similarity docs can be obtained at
`https://labs.wellcomecollection.org/palette-similarity/docs` or
`https://labs.wellcomecollection.org/palette-similarity/redoc`
