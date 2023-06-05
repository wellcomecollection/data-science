# CLIP search

Visual semantic search using OpenAI's [CLIP model](https://openai.com/research/clip).

## Running locally

Build the containers with `docker compose build`.

Then, to start the API and webapp containers, run:

```bash
docker compose up webapp
```

## Components

### Pipeline

Re-run the pipeline into a new index with:

```bash
docker compose run pipeline
```

### API

Build the infrastructure for deployment by navigating to the terraform directory and running

```bash
terraform init
terraform apply
```

Then log in to ECR with:

```bash
aws ecr --profile data-dev get-login-password | docker login --username AWS --password-stdin 964279923020.dkr.ecr.eu-west-1.amazonaws.com
```

Build a new version of the container with:

```bash
docker build -t clip-search -f api/Dockerfile .
```

If you're running on an M1 mac, you might need to run the following instead:

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t clip-search -f api/Dockerfile .
```

Then tag and push the image to ECR

```bash
docker tag clip-search 964279923020.dkr.ecr.eu-west-1.amazonaws.com/weco/clip-search:latest
docker push 964279923020.dkr.ecr.eu-west-1.amazonaws.com/weco/clip-search:latest
```

Update the service by running

```bash
AWS_PROFILE=data-dev aws ecs update-service \
  --service clip-search \
  --cluster data-science-clip-search \
  --force-new-deployment
```

### Webapp

Deployed on vercel. Deploy a new version with `yarn deploy`.
