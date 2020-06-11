# How to

## Setup

Clone [docker-elk](https://github.com/deviantony/docker-elk) to run a local version of the elastic ELK stack.

Give elasticsearch more memory for handling large hashes by altering the `ES_JAVA_OPTS` in the `docker-elk` repo's `docker-compose.yml`:

```yaml
services:
  elasticsearch:
    environment:
      ES_JAVA_OPTS: "-Xmx1g -Xms1g"
```

run the stack by running `docker-compose up` from the top level of the elk repo.

## Build docker images

run `docker-compose build` to build all the images necessary for this project

## Populate

Populate the ES cluster with hashes by running `docker-compose run populate`

Note: You'll also need the ELK stack running.

## API

Run an API over the top of the ES index by running `docker-compose run api`.

Note: You'll also need the ELK stack running and populated.

## Benchmark

Benchmark the similar image query times by running `docker-compose run benchmark`. When complete, this will print a table of average response times

Note: You'll also need the ELK stack running and populated, and the API running.

## Compare

Compare the subjective similarity of LSH model results by running `docker-compose run compare`. This will run a local react app, presenting two randomly chosen models' results for the same random query image. Your choices of the 'better' model in each case will be recorded in another ES index.

Note: You'll also need the ELK stack running and populated, and the API running.

## Score

Compute the comparative goodness of LSH models by running `docker-compose run score`. This will use the Elo algorithm to score models against one another.

Note: You'll also need the ELK stack running and populated with results from the `compare` module.
