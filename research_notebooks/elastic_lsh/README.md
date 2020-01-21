# Locality Sensitive Hashing for Approximate Nearest Neighbour Search in Elactisearch

They're actively working on it, but elasticsearch is still not very good at working with dense vectors. Exact nearest neighbour calculations using `dense_vector`s and `scripted` queries are prohibitively slow, and there's no accepted method for approximate nearest neighbour indexing or searching.

Inspired by [Adobe Sensei](https://www.elastic.co/blog/image-recognition-and-search-at-adobe-with-elasticsearch-and-sensei), we've been experimenting with [locality sensitive hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) to roughly convert our image feature vectors into a usable form for nearest neighbour searches, using the `more_like_this` query, eg:

```json
{
  "query": {
    "more_like_this": {
      "fields": ["feature_vector.keyword"],
      "like": [
        {
          "_index": "image-similarity",
          "_id": "nn4vjdbq"
        }
      ],
      "min_term_freq": 1
    }
  }
}
```

## Method

We split the `d`-dimensional feature vectors into `n` sections, creating `d/n` feature groups. We then find `m` clusters within those groups using sklearn's implementation of k-means. Vectors are hashed by simply combining the feature-groups and the subclusters as a list of strings, eg.

```
0-14, 1-4, 2-32, 3-4, 4-32, 5-5, 6-44, 7-63, 8-58, 9-60, 10-55, 11-53, 12-49, 13-10 ...
```

With `n` x `m` possible values indexed, we should be providing sufficient chance of collision to make good use of elasticsearch's natural tf-idf search, while giving us sufficient perplexity/variance to push the most similar results to the top of the list.

Having experimented a bit, I think we get a better balance of precision and recall by using `n > m` - we've got our best results so far with 120,000 4096-dimensional vectors using `n=256` and `m=32`, but this can definitely be tuned further through experimentation. It's unclear how these numbers should scale with the number of vectors, but my instinct is to leave `n` fixed and linearly increase `m`.

We've also experimented with clustering correlated features within the vector before finding the sub-clusters within those groups, in the hope of finding stronger signals and therefore more "meaningful" clusters.  
In practice, we find that the results get worse, not better, using this method, having effectively run a poorly thought out dimensionality reduction technique on the data before clustering. We should instead seek to optimise the discovery of meaningful clusters within the groups using more sophisticated algorithms than kmeans, or _minimising_ the correlation between features.

## Example results

TL;DR, this seems to work!

### VGG features

[Two Chelsea Pensioners in a garden in winter](https://wellcomecollection.org/works/nn4vjdbq) returns these images
![download](https://user-images.githubusercontent.com/11006680/72805390-a57e8480-3c4a-11ea-940b-32564954267c.png)

and [Saint Lucy (13th December)](https://wellcomecollection.org/works/s64t5mdy) returns
![download-1](https://user-images.githubusercontent.com/11006680/72805388-a4e5ee00-3c4a-11ea-9319-0b3f96cd8ceb.png)

### Palette embeddings

[Two anatomical oil paintings by D'Agoty](https://wellcomecollection.org/works/aamhp58q) returns these
![download-1 (1)](https://user-images.githubusercontent.com/11006680/72805389-a4e5ee00-3c4a-11ea-8a3b-975082a8fcc3.png)
