# Image Pathways

Using feature vectors from the Collection images, we find pathways between images.

This work is inspired by [Google's X Degrees of Separation](https://artsexperiments.withgoogle.com/xdegrees/ogGvLdZg_9FlIQ/rgFS4R9EBpvf5Q).


# Description of the notebooks

Notebooks 1, 2 and 7 are where we get and save the image data. In notebooks 3, 4, 5, 6 we experiment with making pathways - using different pathway algorithms whilst using graphs or not using graphs, and using reduced feature vectors or not. Finally in notebook 8 we look at pathways whilst using the pathway algorithm we decided was best.


## 1. Preprocess_images.ipynb

In this notebook we
- Take all 120588 images from the S3 bucket `wellcomecollection-miro-images-public`.
- We read the image, resize it to having a maximum length or width of 224, and convert any black and white images to RGB.
- These images are then saved in batches to `storage/data`, with the prefix 'processed_images_batch_'.
- We also take a look at how big all the images were, in which we found some of the images were so big that it was worth not saving them in batches, but rather each in a separate png file, which would make reading easier. Thus, we read the previous batches of processed images and saved each image as a png, e.g. A0000001.png
- Due to some image errors 4 of the batches didn't work, so the images from these are saved individually in pngs at the end.
- `storage/data` now contains 120576 images as pngs

Note: 11 images didn't save, the names of these are saved in "../data/images_not_saved" and are:
['B0008000/B0008543.jpg','B0008000/B0008573.jpg','B0009000/B0009632.jpg','B0010000/B0010992.jpg','L0038000/L0038247.jpg',
 'L0083000/L0083878.jpg','L0086000/L0086135.jpg','L0086000/L0086136.jpg','Large Files/L0078598.jpg','Large Files/L0080109.jpg','Large Files/L0080110.jpg']

## 2. Get_feature_vectors.ipynb

In this notebook we:
- Load image names from the data pngs saved in `1. Preprocess_images.ipynb`, removing any which have already had feature vectors found (in the S3 'miro-images-feature-vectors' bucket). This step was neccessary since we ran this code over different sessions.
- Create a dataset, run dataloader and get feature vectors using the vgg16 pretrained model.
- Each feature vector for each image in stored in "feature_vectors/A0000001"
- We then pull in the feature vectors found in the above step, scale them, take a sample of 5000, and use the elbow method to see how many principle components you can reduce to whilst keeping the explained variance at 1. This value is about 100 components.
- We then save dimensionality reduced feature vectors to S3 for these 5000 images, choosing 2, 20, 80, 100, 500, 1000 components. Also saved in the 'miro-images-feature-vectors' bucket under the prefixes "reduced_feature_vectors_i_dims/A0000001" where i = 2, 20, 80, 100, 500, 1000.
- 120576 images had feature vectors and reduced feature vectors found.

Note:
- If using an instance with a GPU, this notebook will run using the GPU.

## 3. Graph_pathways.ipynb

In this notebook we:
- Get the distances between each feature vector (using a sample)
- Create graphs with different types of distance matrices (whether you use a top n neighbour approach or a cosine distance threshold)
- Get the dijkstra_path between 2 random nodes using G_top, G_threshold and G_top_threshold networks
- Plot the umap reduced plot of images with the path shown

Note:
- This is an interim experimental notebook - most of the more thorough experiments are done in `4. Graph_pathways_comparison.ipynb`.

## 4. Graph_pathways_experiments.ipynb

In this notebook we:
- Download the feature vectors/reduced dim feature vectors from S3 (7 options)
- Get the distance matrices for each of these
- Build the graphs using 3 types of neighbour definitions (top neighbours, or neighbours close defined by a threshold, a mixture of these or a fully connected graph)
- Pick the nodes you are going to go between in the network (furthest apart or random?)
- Run different pathways (dijkstra path, the a* path or my defined path) using these graphs

The outcome of this notebook points to using the __raw feature vectors, and with a network where each node is connected to its top 3 neighbours__.

## 5. Graph_pathways_focused.ipynb

In this notebook we:
- Use a graph found using the raw feature vectors, and where each node is connected to its top 3 neighbours, to explore some options of using the dijkstra path, the a* path or my defined path.
- Plot pathways on image plot

## 6. Spaced_pathway.ipynb

In this notebook we:
- Load feature vectors
- Use fv_spaced_pathway_nD to find the pathways in reduced data using different umap.UMAP parameters to reduce the data
- Use fv_spaced_pathway_nD to find the pathways in original feature vectors

We see that just using the original feature vectors to make paths using the `fv_spaced_pathway_nD` function works nicely.

## 7. Save_all_FV_pathways.ipynb

In this notebook we:
- Load all the feature vectors from S3
- Save them in .npy form

## 8. Use_all_FV.ipynb

In this notebook we:
- Load all the feature vectors from the data folder
- Make some pathways using `get_pathway` which is a optimised version of `fv_spaced_pathway_nD` and is the function used in the API code
- Have a look at some pairs of images with a range of different distances between
- Look at a sample of the images plotted in the reduced feature space, and plot a pathway


## Folder structure

```
.
+-- data
|   +-- Where the processed images, and the loaded feature vectors and ids, are stored.
+-- notebooks
|   +-- `1. Preprocess_images.ipynb`: described above ...
|   +-- etc ...
+-- src
|   +-- `network_functions.py`: Where we keep all the functions needed to make graphs, and plot networks and 
         pathways.
```