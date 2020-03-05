# Infer feature vectors

Use trained LSH models to infer feature vector strings for images

```
docker build -t feature_extraction_api .
docker run -v ~/.aws:/root/.aws -p 80:80 feature_extraction_api
```

```
curl "http://0.0.0.0/feature-vector/?image_url=SOME_ENCODED_IMAGE_URL"
```
