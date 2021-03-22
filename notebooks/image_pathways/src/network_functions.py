"""
The functions neccessary for loading, processing and making network graphs and plotting relationships
"""

from tqdm import tqdm
import os
from io import BytesIO
import ast 
import numpy as np
import pickle
from itertools import compress
from collections import Counter
import operator

from PIL import Image
import torch
import boto3
from scipy.spatial.distance import cdist
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from itertools import combinations
import umap.umap_ as umap


def get_all_s3_keys(bucket, s3):
    """
    https://alexwlchan.net/2017/07/listing-s3-keys/
    Get a list of all keys in an S3 bucket.
    """
    keys = []

    kwargs = {'Bucket': bucket}
    while True:
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            keys.append(obj['Key'])

        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break

    return keys

def import_feature_vectors(s3, bucket_name, folder_name, image_name_list):
    
    no_image_name_list = []
    feature_vectors = {}
    for image_name in tqdm(image_name_list):
        try:
            obj = s3.get_object(
                    Bucket=bucket_name,
                    Key=folder_name + "/" + image_name
                )
            read_obj = obj['Body'].read()

            feature_vectors[image_name] = np.frombuffer(
                                    read_obj, dtype=np.float32
                                    )
        except:
            no_image_name_list.append(image_name)
    
    return feature_vectors, no_image_name_list


def get_distances(feature_vectors):
    """
    Get the cosine distance between each image feature vector
    Low cosine distance = similar image features
    """
    feature_vectors_list = list(feature_vectors.values())
    
    dist_mat = cdist(feature_vectors_list, feature_vectors_list, metric="cosine")
    
    return dist_mat

def get_top_neighbours(dist_mat, n):
    """
    Return a distance matrix between nodes
    where there are only distances given 
    for the n lowest cosine neighbours of each node.
    'None' signifies that the nodes are not
    connected.
    """
    
    dist_mat_top = np.zeros_like(dist_mat)
    dist_mat_top[:]=None

    # Find the top n neighbours for each image
    for i in tqdm(range(0,len(dist_mat))):
        arr = dist_mat[i].argsort()
        top_args = arr[arr!=i]
        dist_mat_top[i][top_args[0:n]] = dist_mat[i][top_args[0:n]]
        for j in top_args[0:n]:
            dist_mat_top[j][i] = dist_mat[j][i]
            
    return dist_mat_top

def get_high_neighbours(dist_mat, dist_threshold):
    """
    Return a distance matrix between nodes
    where there are only distances given 
    for distances under a threshold.
    'None' signifies that the nodes are not
    connected.
    """
    
    dist_mat_top = np.zeros_like(dist_mat)
    dist_mat_top[:]=None

    i_threshold = np.where(dist_mat < dist_threshold)
    if len(i_threshold)!=0:
        for (i,j) in zip(i_threshold[0], i_threshold[1]):
            if i!=j:
                dist_mat_top[i][j] = dist_mat[i][j]
    else:
        return print("No distances less than this threshold")

    return dist_mat_top

def get_top_high_neighbours(dist_mat, n, dist_threshold):
    """
    Return a distance matrix between nodes
    where there are only distances given 
    for distances under a threshold, or the top n
    highest distances.
    'None' signifies that the nodes are not
    connected.
    """
    
    dist_mat_top = np.zeros_like(dist_mat)
    dist_mat_top[:] = None

    for i in tqdm(range(0, len(dist_mat))):
        arr = dist_mat[i].argsort()
        top_args = arr[arr!=i]
        
        thresh_args = top_args[dist_mat[i][top_args] < dist_threshold]
        n_args = top_args[0:n]
        best_args = np.unique(np.concatenate((thresh_args, n_args), 0))
        
        dist_mat_top[i][best_args] = dist_mat[i][best_args]
        for j in best_args:
            dist_mat_top[j][i] = dist_mat[j][i]

    return dist_mat_top

def inv_rel_norm(value, min_val, max_val):
    value = (value - min_val)/(max_val - min_val)
    value = 1/(value+1e-8)
    return value

def create_graph(dist_mat_top):
    
    min_val = np.nanmin(dist_mat_top)
    max_val = np.nanmax(dist_mat_top)
    
    nodes = list(range(0,len(dist_mat_top[0])))

    G = nx.Graph()
    G.add_nodes_from(nodes)

    # Put the weights in as the distances
    # only inc nodes if they are in the closest related neighbours
    for start, end in list(combinations(nodes, 2)):
        if ~np.isnan(dist_mat_top[start, end]):
            # Since in the plot a higher weight makes the nodes closer, 
            # but a higher value in the distance matrix means the images are further away,
            # we need to inverse the weight (so higher = closer)
            G.add_edge(
                start,
                end,
                weight=inv_rel_norm(dist_mat_top[start, end], min_val, max_val)
            )
    return G

def plot_graph(G, figsize=(10,10), image_names=None, node_list=None, pos=None):
    """
    pos: if you want to input some already found positions, say from a previous plot
        then you can do so
    """
    
    if not pos:
        pos = nx.spring_layout(G)

    plt.figure(3, figsize) 
    
    if node_list:
        nx.draw(G, pos, alpha = 0)
        nx.draw_networkx_nodes(G, pos,
                               node_color='r', alpha = 0.2,
                               node_size=figsize[0]*4)
        nx.draw_networkx_nodes(G, pos,
                               nodelist=node_list,
                               node_color='b', node_size=figsize[0]*4)
        
        edge_list = [(v, node_list[i + 1] )  
                 for i, v in enumerate(node_list[:-1])]
        nx.draw_networkx_edges(G, pos,
                           edge_color='r', alpha = 0.2,)
        nx.draw_networkx_edges(G, pos,
                               edgelist=edge_list,
                               edge_color='b')
    else:
        nx.draw(G, pos, node_size=figsize[0])
         
    if image_names:
        for p in pos:  # raise text positions
            pos[p][1] += 0.06
        
        image_names_dict = {k:str(k)+" "+v for k,v in enumerate(image_names)}
        nx.draw_networkx_labels(G, pos, labels=image_names_dict)
    
    plt.show()
    
    return pos
    
def create_network_graph(dist_mat_top):
    G = create_graph(dist_mat_top)
    return G

def visualise_clusters(feature_vectors_list):
    
    reducer = umap.UMAP()
    embedding_fv = reducer.fit_transform(feature_vectors_list)
    embedding_fv.shape

    x_data = [[a, b] for (a,b) in zip(embedding_fv[:, 0], embedding_fv[:, 1])]

    visualize_scatter_with_images(
        x_data,
        images = images,
        image_zoom=0.3)
    
def reduce_data_nd(feature_vectors, n_components=2, n_neighbors=15, min_dist=0.1):
    
    reducer = umap.UMAP(
        n_components = n_components,
        n_neighbors = n_neighbors,
        min_dist = min_dist
        )
    embedding_fv = reducer.fit_transform(list(feature_vectors.values()))

    x_data = {
            k:v for (k,v) in zip(
                feature_vectors.keys(),
                embedding_fv[:, 0:n_components].tolist()
                )
            }
    
    return x_data

def get_random_node_path(G, image_names_dict):
    
    node1 = np.random.choice(list(image_names_dict))
    node2 = np.random.choice(list(image_names_dict))

    node_path = nx.dijkstra_path(G, node1, node2, weight=None)
    
    image_names_path = [image_names_dict[n] for n in node_path]

    return image_names_path
        
        
def image_pathway_plot(images_dir, image_type, node_path, title=None):

    fig = plt.figure(figsize=(20,10))
    columns = len(node_path)
    for i, image_name in enumerate(node_path):
        image = Image.open(images_dir + image_name + image_type)
        ax = plt.subplot(2, columns, i + 1)
        if title and i==0:
            plt.title(title)
        ax.set_axis_off()
        plt.imshow(image)
        image.close()
        
def image_pathway_scaled_plot(images_dir, image_type, node_path, title=None):
    
    plot_images = []
    plot_images_sizes =[]
    for image_name in node_path:
        image = Image.open(images_dir + image_name + image_type)
        plot_images.append(image)
        plot_images_sizes.append(image.size)
        
    max_y = max([c[1] for c in plot_images_sizes])
    rescale_x = [c[0]*max_y/c[1] for c in plot_images_sizes]
    
    columns = len(node_path)

    fig = plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(1, columns, width_ratios=rescale_x) 

    for i, image in enumerate(plot_images):
        ax = plt.subplot(gs[i])

        if title and i==0:
            plt.title(title)
        ax.set_axis_off()

        plt.imshow(image)
    

def visualize_scatter_with_images(X_2d_data, image_name_list, images_dir, image_type, figsize=(45,45), image_zoom=1, pathway=None):
    """
    from https://www.kaggle.com/gaborvecsei/plants-t-sne
    """
    
    if pathway:
        x_path = [X_2d_data[c][0] for c in pathway]
        y_path = [X_2d_data[c][1] for c in pathway]
    
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, image_name in zip(X_2d_data.values(), image_name_list):
        try:
            image = Image.open(images_dir + image_name + image_type)
            x0, y0 = xy

            if pathway:
                img = OffsetImage(image, zoom=image_zoom, alpha = 0.1)
            else:
                img = OffsetImage(image, zoom=image_zoom)
            image.close()
            ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        except:
            print("Image {} not found".format(image_name))
            
    # Print the pathway images after these since otherwise they get hidden
    if pathway:
        for xy, image_name in zip([X_2d_data[n] for n in pathway], pathway):
            image = Image.open(images_dir + image_name + image_type)
            x0, y0 = xy
            img = OffsetImage(image, zoom=image_zoom*2, alpha = 1)
            image.close()
            ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
    
    ax.update_datalim(list(X_2d_data.values()))
    ax.autoscale()
    if pathway:
        plt.plot(x_path, y_path, 'ro-', linewidth=5)
    plt.axis('off')
    plt.show()


def normalise_list(a):
    amin, amax = min(a), max(a)
    a = [(val-amin) / (amax-amin) for val in a]
    return a

def defined_path(G, source, target, G_weights, path_size, best_path=True, best_type="sum"):
    
    """Returns a list of nodes in a path between source and target.

    There may be more than one path.  This returns only one.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    G_weights : A numpy matrix
        A matrix of all the weights between everynode
        in the network.

    path_size : int
        The number of nodes in the path which you would like
        to return.
    
    best_path : boolean
        Whether you want to return the best (True)
        or worst (False) path. Best and worst are
        defined by the path having high weights
        and low weights respectively.
    
    best_type : "sum" or "average" or "variance" or "sumvar"
        Whether you want the path returned to have the 
        highest or lowest (depending on "best_path")
        total (sum) or average of all the weights in the
        pathway, or the smallest variance in pathway
        weights (variance), 
        or a mixture of low variance and high sum of 
        pathway weights (sumvar)
        
    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> G_weights = nx.to_numpy_matrix(G)
    >>> print(defined_path(G, 0, 4, G_weights, path_size=5))
    [0, 1, 2, 3, 4]
    
    """
    
    # Get all the paths between node 1 and node 2 with max this path size
    all_paths = []
    for path in nx.all_simple_paths(G, source, target, cutoff = (path_size-1)):
        all_paths.append(path)
        
    if all_paths == []:
        return print(
            "There are no pathways between these",
            "nodes with a path size <= " + str(path_size)
        )

    # Remove paths which are not either your inputted path_size, or the biggest possible size
    len_paths = [len(p) for p in all_paths]
    max_path_size = max(path_size, max(len_paths))
    all_paths = [p for p in all_paths if len(p)==max_path_size]
    print("There are {} path(s) of this size".format(len(all_paths)))
    
    # Return the path which has the highest/lowest sum/average
    # of the weights between each node in the path

    path_summary = {}
    path_vars = {}
    for path_num, path_list in tqdm(enumerate(all_paths)):

        path_edges = Counter([(v, path_list[i + 1] )  
                 for i, v in enumerate(path_list[:-1])])
        weights = []
        for (n1, n2), count in path_edges.items():
            weights.append(G_weights[n1].item(n2)*count)

        if best_type == "sum":
            path_summary[path_num] = sum(weights)
        elif best_type == "average":
            path_summary[path_num] = sum(weights)/max_path_size
        elif best_type == "sumvar":
            # inverse so that larger value is good
            path_summary[path_num] = sum(weights)
            path_vars[path_num] = 1/np.var(weights)
        elif best_type == "variance":
            # inverse so that larger value is good
            path_summary[path_num] = 1/np.var(weights)     
        else:
            return print(
                "Warning: best_type argument not recognised,",
                "please choose 'sum', 'average', 'sumvar' or 'variance'"
            )

    if best_type == "sumvar":
        # Need to normalise both the sum and the variance of 
        # the weights and multiple (otherwise overly weighted
        # by the sum)
        norm_path_summary = normalise_list(list(path_summary.values()))
        norm_path_vars = normalise_list(list(path_vars.values()))
        
        path_summary = {k: w*v for (k, w, v) in zip(path_summary.keys(), norm_path_summary, norm_path_vars)}
        
    if best_path:
        # higher weight = closer
        paths = max(path_summary.items(), key=operator.itemgetter(1))
    else:
        paths = min(path_summary.items(), key=operator.itemgetter(1))


    # If there are multiple paths with the same sum/average weight,
    # then just take first one
    # (maybe in the future we can do something else here)
    path = all_paths[paths[0]]
    
    return path


def fv_spaced_pathway_nD(x_data, node1, node2, n_nodes):
    
    """
    Parameters
    ----------
    x_data : nD coordinates of images from reduced feature vectors, 
        or just the plain feature vectors
        Can be found using: 
        x_data = reduce_data(feature_vectors)
        or
        x_data = {k:list(v) for k,v in feature_vectors.items()}

    node1 : node number
       Starting node for path

    node2 : node number
       Ending node for path
       
    n_nodes : number of nodes wanted in this pathway
       
    Examples
    --------
    >>> x_data = reduce_data(feature_vectors)
    >>> node_path = fv_spaced_pathway(x_data, 106, 35, 4)
    >>> node_path
    ['L0043469', 'L0076702', 'V0035626ER', 'V0042397']
    
    """
    n_components = len(list(x_data.values())[0])

    nodes = np.array(list(x_data.values()))

    start_coord = nodes[node1]
    end_coord = nodes[node2]
    
    # Get the equation of the line
    line_vect = end_coord - start_coord
    y_line = lambda t: start_coord + t*line_vect

    t_spaced = np.linspace(0, 1, num = (n_nodes-1), endpoint = False)[1:]
    y_spaced = [y_line(x) for x in t_spaced]
    
    nodes = np.delete(nodes, [node1, node2], axis = 0)
    
    node_path = []
    previous_coord = []
    for i, node in enumerate(y_spaced):
        
        # Don't include the nodes that are
        # behind the plane that the previous node
        # sits in, and that is perpendicular to 'node'
        # [The plane perpendicular to (𝑎,𝑏,𝑐) and passing through (𝑥0,𝑦0,𝑧0) is
        # 𝑎(𝑥−𝑥0)+𝑏(𝑦−𝑦0)+𝑐(𝑧−𝑧0)=0, so if this is >0 it should be 'in front' of the plane]

        #if len(previous_coord) != 0:
            #plane_values = np.sum((nodes - previous_coord)*line_vect, axis = 1)
            #nodes = nodes[plane_values > 0] 
            
        dist = np.sum((nodes - node)**2, axis=1)
        node_arg = np.argmin(dist) 
    
        closest_coord = nodes[node_arg]
        
        previous_coord = closest_coord
        
        node_path.append(list(closest_coord))
        
        # Take this node out of the list of nodes so that you can't pick it again
        nodes = np.delete(nodes, node_arg, axis = 0)

    node_path.insert(0, list(start_coord))
    node_path.extend([list(end_coord)])
    
    # Convert to image names rather than coordinates
    node_path = [list(x_data.keys())[list(x_data.values()).index(n)] for n in node_path]
    return node_path


def get_pathway(feature_vectors_ids, feature_vectors, id_1, id_2, n_nodes, sample_size):
    
    """
    Parameters
    ----------
    feature_vectors_ids: numpy array
        The filenames of the images that each feature vector
        corresponds to

    feature_vectors : numpy array
        Feature vectors of images (each image is 4096x1)

    id_1 : image name
        Starting node for path

    id_2 : image name
       Ending node for path
       
    n_nodes : int
        Number of nodes wanted in this pathway

    sample_size : int
        To make this quicker use a sample
        of the images, if None then all
        the feature vectors will be used

    Returns
    -------

    node_path : list of strings
        A list of the image names in the pathway

    Examples
    --------
    >>> node_path = get_pathway(feature_vectors_ids, feature_vectors, 'L0043469', 'V0042397', 4, None)
    >>> node_path
    ['L0043469', 'L0076702', 'V0035626ER', 'V0042397']
    
    """

    n_components = len(list(feature_vectors)[0])

    # Get the node numbers
    node1 = list(feature_vectors_ids).index(id_1)
    node2 = list(feature_vectors_ids).index(id_2)

    start_coord = feature_vectors[node1]
    end_coord = feature_vectors[node2]

    # Get the equation of the ideal line
    line_vect = end_coord - start_coord
    ideal_line = lambda t: start_coord + t*line_vect

    t_spaced = np.linspace(0, 1, num = (n_nodes-1), endpoint = False)[1:]
    ideal_vectors = np.array([ideal_line(x) for x in t_spaced])

    feature_vectors = np.delete(feature_vectors, [node1, node2], axis = 0)
    feature_vectors_ids = np.delete(feature_vectors_ids, [node1, node2], axis = 0)
    # Sample the remaining feature vectors
    # I want the user to be able to input any image they like, 
    # so I only use the sample for the pathway images

    if sample_size:
        
        rand_args = np.random.choice(
                        range(0, len(feature_vectors)),
                        int(sample_size)-2,
                        replace = False
                    )
        feature_vectors = feature_vectors[rand_args]
        feature_vectors_ids = feature_vectors_ids[rand_args]

    dists = cdist(feature_vectors, ideal_vectors)
    argsorted = np.argsort(dists, axis = 0).T

    path_dists = []
    path_indexes = []
    for row in argsorted:
        i, found_node = 0, False
        while not found_node:
            if row[i] not in path_indexes:
                path_indexes.append(row[i])
                path_dists.append(dists[row[i]])
                found_node = True
            else: 
                i += 1 

    node_path = [feature_vectors_ids[index] for index in path_indexes]
        
    node_path.insert(0, id_1)
    node_path.extend([id_2])
    
    path_dists = [p[i] for i, p in enumerate(path_dists)]

    return node_path, path_dists
