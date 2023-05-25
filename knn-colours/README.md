# Indexing colours for knn queries

Elasticsearch introduced [knn queries](https://www.elastic.co/guide/en/elasticsearch/reference/master/knn-search.html) in [version 8](https://www.elastic.co/blog/introducing-approximate-nearest-neighbor-search-in-elasticsearch-8-0). 

If we're thoughtful about how we index our inferred data about images, this might significantly improve our colour search / filtering / similarity functionality.
