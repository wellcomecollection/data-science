# Simple viewer for testing local inference

This is a tool for testing the results of locally run inference when developing models for image similarity. It is designed to be used with the scripts in the parent directory, and not to be opinionated about the model in question.

![image](https://user-images.githubusercontent.com/4429247/105062744-d9190800-5a72-11eb-8d4e-16f49bc26176.png)

## Usage

*Before starting*: download some images using `../fetch_images_from_api.py`. Run them through a locally running inferrer server using `../run_inference.py`. Index the resultant JSON file into a locally running Elasticsearch server using `../index_inferrer_output.py`.

1. Start an image server which will be able to serve images at the path `/[id].jpg` - for example, run `python3 -m http.server 3333` in the directory containing the images you downloaded to start such a server on port 3333.
2. If the config in `.env` is insufficient, override it in a `.env.local` file.
3. Start the viewer with `yarn dev`.
4. In the viewer interface, select the index you want to use and enter the name of the field which you want to use for `more_like_this` similarity. Enter the number of similar images you want to search for, and click go to fetch a random set of images.

![image](https://user-images.githubusercontent.com/4429247/105062286-57c17580-5a72-11eb-9f33-aac8117b4616.png)

## Calculated field values

If you want to be able to calculate a field value to use for similarity searches (eg by creating a hash for a given colour), add a module to `modules` which default-exports a function like `string => string | string[]`, and register it in `modules/index.ts`. It will then appear in the dropdown for `Calculated field value`:

![image](https://user-images.githubusercontent.com/4429247/105062672-c3a3de00-5a72-11eb-88b0-18f806651c41.png)

