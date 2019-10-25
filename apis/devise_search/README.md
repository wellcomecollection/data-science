# DeViSE search

Connects visual features in images to semantic features in query text, allowing users to explore uncaptioned images using regular text queries. Based on the 2012 paper [DeViSE: A Deep Visual-Semantic Embedding Model](https://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model.pdf) and more recent work on language modelling and sentence embeddings.

## How it works

Query text is fed into a pytorch query-embedding model, producing a 4096 dimensional array. The model tries to place the query in a region of feature space which matches the feature space of images, ie. an embedding of the word `'dog'` will approximately match the the embedding produced by VGG16 when presented with a picture of a dog. Having trained a model on many examples of image-caption pairs, We remove the need for captions and can approximate captionless image search based on visual features alone.

Feature vectors for all collection images are preprocessed into an index by [nmslib](https://github.com/nmslib/nmslib), allowing for fast nearest-neighbour matching after the query-embedding has been generated.

## Output example

Hitting [`https://labs.wellcomecollection.org/devise_search`](https://labs.wellcomecollection.org/devise_search) will use `'An old wooden boat'` as an example query, returning:

```json
{
  "query_text": "An old wooden boat",
  "neighbour_ids": [
    "V0047059",
    "L0011115",
    "V0047043",
    "V0047044",
    "N0022525",
    "V0047061",
    "M0000981",
    "L0012178",
    "V0045571EL",
    "V0047063"
  ],
  "neighbour_urls": [
    "https://iiif.wellcomecollection.org/image/V0047059.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0011115.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0047043.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0047044.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/N0022525.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0047061.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/M0000981.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/L0012178.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0045571EL.jpg/full/960,/0/default.jpg",
    "https://iiif.wellcomecollection.org/image/V0047063.jpg/full/960,/0/default.jpg"
  ]
}
```

The API accepts a `query_text` parameter. More detailed API docs can be obtained at `https://labs.wellcomecollection.org/devise_search/docs` or `https://labs.wellcomecollection.org/devise_search/redoc`
