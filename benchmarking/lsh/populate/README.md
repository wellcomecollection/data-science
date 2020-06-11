# Populate

Trains a model with a specified set of parameters (`n_classifiers`, `n_components`), transforms a full set of image feature data into the trained model hash space, and indexes those hashes into an elasticsearch index with a corresponding name.

`cli.py` allows the user to specifiy the parameters, while `populate.py` runs the process for a wide range of parameter space and indexes the exact features as well.
