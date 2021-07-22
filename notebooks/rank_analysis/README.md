# Analysis notebooks

We want to track some broad metrics on search performance using real user data.

Benchmarking the performance of search on a consistent set of search terms wil blah blah blah.

## Running these notebooks locally

If you don't want to go through the process of installing jupyter, resolving environments/requirements etc, just run `docker compose up --build` to build the analysis project in a container. This should start a jupyter lab instance locally with all the dependencies installed, and environment variables pulled in from the parent directory. If you haven't fetched a `.env` from vercel already, follow the instructions at the root of this repo.
