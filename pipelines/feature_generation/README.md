# Generate feature index
## make sure you have the requisite data
you should download a catalogue snapshot
```
wget https://data.wellcomecollection.org/catalogue/v2/works.json.gz
gunzip works.json.gz
```

and the full set of feature vectors from the relevant s3 bucket
```
aws s3 sync s3://miro-images-feature-vectors /Users/pimh/Desktop/feature_vectors/ --profile data-dev
```

## do the zipping together of data and building of the nmslib index
```
python generate_index.py --works_json_path /path/to/works.json --feature_vector_dir /path/to/feature_vectors
```

