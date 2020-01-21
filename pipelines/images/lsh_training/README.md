# LSH (Locality Sensitive Hashing) Training

Uses the features from [feature_extraction](../feature_extraction) to train a series of k-means clustering algorithms, which are then used in [lsh_inference](../lsh_inference) to infer a set of hashes for individual feature vectors, before they're sent to elasticsearch.

Written as generically as possible so that we can use the generic structure for any set of feature vectors
