# Semantic search

Prototype semantic search api and webapp for wellcomecollection content.

## Running locally

To start the API and webapp containers, run:

```bash
docker compose up --build webapp
```

## Deploying

### API

The API is deployed on AWS ECS, using the terraform in the `terraform` directory. Follow the instructions in `api.README.html` to update the deployment.

To build the image, run the following from the project root:

```bash
docker build -t semantic-search -f api/Dockerfile .
```

If you're running on an M1 mac, you might need to run the following instead:

```bash
docker buildx build --push --platform linux/amd64,linux/arm64 -t semantic-search -f api/Dockerfile .
```

To log in to ECR, you might need to swap the suggested command with something like this:

```bash
aws ecr --profile data-dev get-login-password | docker login --username AWS --password-stdin 964279923020.dkr.ecr.eu-west-1.amazonaws.com
```

### Webapp

The webapp is deployed on vercel. To update the deployment, push to the `main` branch, or run `yarn run vercel --prod` locally.
