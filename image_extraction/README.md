# Image extraction

We want to extract more 'interesting' images from our digitised books, archives and manuscripts.

## Project structure

`docs` contains notes made during the course of this research.

This project is divided into a few component docker containers which all serve different functions. They're coordinated into a complete set of services by the `docker-compose.yml` file.

- **get_images**: Download all the images from a work
