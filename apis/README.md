# Experimental APIs

- **[Palette similarity:](./palette_similarity)** Uses 5-colour palettes for
  each image to find most similarly coloured images to a query image.

- **[Image similarity:](./image_similarity)** Uses VGG16 feature vectors to find
  visually similar images to a query image.

- **[Image pathways](./image_pathways)** Uses the feature vectors above to
  construct visually coherent pathways in feature-space between two images.

- **[Visual-semantic search:](./devise_search)** Connects visual features in
  images to semantic features in query text, allowing users to explore
  uncaptioned images using regular text queries.

- **[Named entity recognition & disambiguation:](./nerd)** Annotates query text
  with disambiguated subjects and entities using the wikipedia/wikidata
  knowledge base.
