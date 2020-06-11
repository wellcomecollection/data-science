# Benchmark

`cli.py` allows the user to benchmark a model's query speed, eg:

```
python cli.py n_classifiers=32 n_clusters=32 sample_size=10000
```

will run 10000 random similar images queries for the specified model and record the response time for each one, returning the mean time.

`benchmark_lsh.py` will do the same for every available model.
