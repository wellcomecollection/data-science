# Image similarity

Uses VGG16 feature vectors to find visually similar images to a query image.

## How it works

4096-dimensional feature vectors are obtained for all images in the collection using a pretrained VGG16 pytorch model, before being indexed into an [nmslib](https://github.com/nmslib/nmslib) object, allowing for fast nearest-neighbour matching.

## Output example

Hitting
[`https://labs.wellcomecollection.org/feature-similarity/works/{catalogue_id}`](https://labs.wellcomecollection.org/feature-similarity/works/pp6f97px)
will return the `n=10` nearest neighbours to the given image (`catalogue_id`) in the
form:

```json
{
  "original": {
    "miro_id": "L0008006",
    "catalogue_id": "pp6f97px",
    "miro_uri": "https://iiif.wellcomecollection.org/image/L0008006.jpg/full/960,/0/default.jpg",
    "catalogue_uri": "https://wellcomecollection.org/works/pp6f97px"
  },
  "neighbours": [
    {
      "miro_id": "L0020766",
      "catalogue_id": "qg9wwhke",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0020766.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/qg9wwhke"
    },
    {
      "miro_id": "M0009109",
      "catalogue_id": "cnp6ruqa",
      "miro_uri": "https://iiif.wellcomecollection.org/image/M0009109.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/cnp6ruqa"
    },
    {
      "miro_id": "M0016741",
      "catalogue_id": "wtags2zd",
      "miro_uri": "https://iiif.wellcomecollection.org/image/M0016741.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/wtags2zd"
    },
    {
      "miro_id": "V0002881",
      "catalogue_id": "fkd5a9dk",
      "miro_uri": "https://iiif.wellcomecollection.org/image/V0002881.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/fkd5a9dk"
    },
    {
      "miro_id": "M0000267",
      "catalogue_id": "jrv6e6ja",
      "miro_uri": "https://iiif.wellcomecollection.org/image/M0000267.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/jrv6e6ja"
    },
    {
      "miro_id": "L0027395",
      "catalogue_id": "fvpkmb8w",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0027395.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/fvpkmb8w"
    },
    {
      "miro_id": "L0036005",
      "catalogue_id": "whd3m9t2",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0036005.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/whd3m9t2"
    },
    {
      "miro_id": "L0000787",
      "catalogue_id": "zbnmmufv",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0000787.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/zbnmmufv"
    },
    {
      "miro_id": "M0003472",
      "catalogue_id": "t7hwhd4v",
      "miro_uri": "https://iiif.wellcomecollection.org/image/M0003472.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/t7hwhd4v"
    },
    {
      "miro_id": "L0015988",
      "catalogue_id": "th87hqvk",
      "miro_uri": "https://iiif.wellcomecollection.org/image/L0015988.jpg/full/960,/0/default.jpg",
      "catalogue_uri": "https://wellcomecollection.org/works/th87hqvk"
    }
  ]
}
```
