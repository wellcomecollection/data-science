# Palette similarity

Uses 5-colour palettes for each image to find most similarly coloured images to a query image.

## How it works

Query text is fed into a large backbone network (some kind of transformer/RNN based architecture) which generates meaningful embeddings for each token. Two heads then share this embedding:

- The first predicts whether the token is part of an entity name, producing a binary decision for each token.
- The second predicts the region of wikidata-space which is being referred to by the token and its surroundings. Predicting a region of wikidata-space is made possible by training on embeddings from [PyTorch BigGraph](https://github.com/facebookresearch/PyTorch-BigGraph), which creates graph embeddings of the wikidata entity hierarchy.

Spans of successive positive entity labels are aggregated and a query for the complete span is sent to the [wikipedia api](https://en.wikipedia.org/w/api.php?action=query&format=json&redirects&prop=pageprops&titles=something), returning a list of plausible candidates.

The order in which the candiates are returned by the wikipedia API (ie. wikipedia's best guess at which entity is being referred to) is combined with the distance between the model's predicted embedding and the already-known embedding of each candidate, and a best match is selected.

This process is intended to (very roughly) approximate the human experience of annotating text in the wikipedia interface.

## Output example

The output shape of this API is still in flux... Will update soon.
