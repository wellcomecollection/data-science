# Benchmarking lsh queries

We're using a [Locality-sensitive Hashing](https://en.m.wikipedia.org/wiki/Locality-sensitive_hashing) (LSH) model to encode image feature vectors into text, allowing them to be searched and compared natively in elasticsearch.

The current parameters we're using for the model give good-enough results, but the response time is too slow.

The code here allows us to train a set of models with a range of parameter values, and then compare their response times and the subjective goodness of the results.

See [howto](howto.md) for a description of how to run the code and [results](results.md) for some tables of results.
