import numpy as np
import tensorflow as tf
import sys
import os
import os.path as osp
import pandas as pd

from functools import partial
from itertools import repeat

sys.path.append('./octnet/py/')
import pyoctnet

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from multiprocessing import Pool

import scipy
from scipy import ndimage
from scipy.sparse import coo_matrix

from octree import octree_func
from util import *
from sklearn import linear_model, datasets
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn import mixture


def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind.astype(int)

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = np.expand_dims((ind.astype('int') / array_shape[1]), axis=-1)
    cols = np.expand_dims(ind % array_shape[1], axis=-1)
    coordinates = np.concatenate([rows, cols], axis =-1)
    return coordinates

def rough_LBS(points):
    ransac = linear_model.RANSACRegressor()
    ransac.fit(points[:,0:2], points[:,2])
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    background_points = points[inlier_mask, 0:3]
    foreground_points = points[outlier_mask, 0:3]
    return foreground_points, background_points

def get_cluster_cellmap_from_raw_points(raw_points, n_threads=16, cluster_mode="kmeans", seeds=[1,22123,3010,4165]):

    # affinity = 'rbf'
    affinity = 'nearest_neighbors'
    num_neighbors = 20
    num_cluster = 64
    # seeds = [1,22123,3010,4165,54329,62,7656,80109212] # for better generalization
     # for better generalization
    cluster_idx = range(num_cluster)
    cellmap = {}
    for s in range(len(seeds)):
        if cluster_mode == "dirichlet":
            spectral = mixture.BayesianGaussianMixture(n_components=num_cluster, covariance_type='full', random_state=seeds[s]).fit(raw_points)
            cluster_idx = np.unique(spectral.predict(raw_points))
        else:
            if cluster_mode == 'kmeans':
                spectral = KMeans(n_clusters=num_cluster, random_state=seeds[s], n_jobs=n_threads).fit(raw_points)
            elif cluster_mode == "spectral":
                spectral = SpectralClustering(num_cluster, random_state=seeds[s], affinity=affinity, n_neighbors=num_neighbors, n_jobs=n_threads).fit(raw_points)
        for i, index in enumerate(cluster_idx):
            if hasattr(spectral, 'labels_'):
                idx = np.where(spectral.labels_.astype(np.int)==index)[0]
            else:
                idx = np.where(spectral.predict(raw_points)==index)[0]
            cellmap[s*num_cluster+i] = raw_points[idx]
    return cellmap

def get_cellmap_from_raw_points(raw_points, n_threads=16, split_mode=0, cluster_mode='kmeans', cell_per_meter=1, cellmap_radius=30):
    
    foreground_points, background_points = rough_LBS(raw_points)
    if split_mode == 0:
        point = foreground_points
    elif split_mode == 1:
        point = background_points
    elif split_mode == 2:
        point = raw_points
    elif split_mode == 3:
        return get_cluster_cellmap_from_raw_points(foreground_points, n_threads, cluster_mode=cluster_mode)
    cellmap_size = 2*cell_per_meter*cellmap_radius
    
    # Extract foreground points within cellmap
    index_within_cellmap = (abs(point[:, 0]) < cellmap_radius) &  (abs(point[:, 1]) < cellmap_radius);
    points_within_cellmap = point[index_within_cellmap, :]
    cell_id_2d = cellmap_radius*cell_per_meter + np.floor(points_within_cellmap[:, 0:2]*cell_per_meter)

    cell_id = sub2ind([cellmap_size, cellmap_size], cell_id_2d[:,0], cell_id_2d[:,1])
    cell_id_unique = np.unique(cell_id) 

    cellmap = {}
    for i_cell in cell_id_unique:
        cell_id_index = (cell_id == i_cell)
        cellmap[i_cell] = points_within_cellmap[cell_id_index, :]

    return cellmap

def get_octree_center(raw_points, meta, octree_depth=2, num_points_lower_threshold=10, num_points_upper_threshold=200):
    centers = []
    for i in range(raw_points.shape[0]):
        point = raw_points[i, range(min(meta[i], num_points_upper_threshold)), :]

        world_size = max(point[:,0].max()-point[:,0].min(), point[:,1].max()-point[:,1].min(), point[:,2].max()-point[:,2].min())
        origin = point.mean(axis=0)
        octree_dict = {}
        point_set = tuple(map(tuple, point))
        octree_func(octree_dict, point_set, num_points_upper_threshold, octree_depth, world_size, origin)

        octree_center = []
        for key, value in octree_dict.items():
            if key[0] == octree_depth-1:
                point_subspace = point[np.asarray(value, dtype=np.int32),:]
                octree_center.append(point_subspace.mean(axis=0))
        octree_center = np.array(octree_center)
        centers.append(octree_center)
    return centers


def get_points_from_cellmap(cellmap, cell_per_meter=1, cellmap_radius=30, num_points_lower_threshold = 50):

    points = None
    for i_cell in cellmap.keys():
        if points == None:
            points = cellmap[i_cell]
        else:
            points = np.concatenate([points, cellmap[i_cell]], axis=0)
    return points


def get_batch_data_from_cellmap(cellmap, num_points_lower_threshold=50, num_points_upper_threshold=200):

    batch_data = []
    number_points = []
    centroids = []

    for i_cell in cellmap.keys():

        points = cellmap[i_cell]
        num_points = points.shape[0]
        dim = points.shape[-1]
        if num_points < num_points_lower_threshold:
            # no AE compression
            continue

        number_points.append(num_points)

        # sampling if too many points
        if num_points > num_points_upper_threshold:
            sample_index = np.random.choice(num_points, num_points_upper_threshold, replace=False)
            points = points[sample_index, :]

        # centralize data
        centroid = points.mean(axis=0)
        centroids.append(centroid)
        points = points - centroid

        # data same shape
        if num_points < num_points_upper_threshold:
            cellpoints = np.concatenate([points, np.zeros((num_points_upper_threshold-num_points, dim))],axis=0)
        else:
            cellpoints = points

        batch_data.append(cellpoints)

    # batch * num_points_upper_threshold * 4
    batch_data = np.stack(batch_data)
    # batch
    number_points = np.stack(number_points)
    # batch
    centroids = np.stack(centroids)

    meta_data = {}
    meta_data['number_points'] = number_points
    meta_data['centroids'] = centroids

    return batch_data, meta_data

def shuffle_data(point_clouds, meta, seed=12345):
    num_examples = point_clouds.shape[0]
    if seed is not None:
        np.random.seed(seed)
    perm = np.arange(num_examples)
    np.random.shuffle(perm)
    return point_clouds[perm], meta[perm]

def rescale(point_clouds, num_points, num_points_upper_threshold=200):
    point_mean = []
    point_scaled = []
    for i in range(point_clouds.shape[0]):
        p_scaled = point_clouds[i, np.arange(min(num_points[i], num_points_upper_threshold)), :]
        x_max, y_max, z_max = np.max(p_scaled, 0)
        x_min, y_min, z_min = np.min(p_scaled, 0)
        x_var, y_var, z_var = max(x_max,abs(x_min)), max(y_max,abs(y_min)), max(z_max,abs(z_min))
        point_clouds[i, np.arange(min(num_points[i], num_points_upper_threshold)), :] = p_scaled / max(x_var, y_var, z_var)
        point_scaled.append(point_clouds[i,:,:])
    point_scaled = np.stack(point_scaled)
    return point_scaled


def load_pc_data(batch_size, train=True, split_mode=3, cluster_mode='kmeans', n_threads=16, shuffle=True):
    date = '2011_09_26'
    if train == True:
        drive = '0001'
    else:
        drive = '0005'
    basedir = '/scratch2/sniu/kitti'
    if split_mode == 3:
        num_points_upper_threshold = 1024
    else:
        num_points_upper_threshold = 200

    dataset = load_dataset(date, drive, basedir)
    # for visualization
    # tracklet_rects, tracklet_types = load_tracklets_for_frames(len(list(dataset.velo)), '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(basedir, date, date, drive))
    # frame = 10 # for visualization
    # display_frame_statistics(dataset, tracklet_rects, tracklet_types, frame)

    dataset_velo = list(dataset.velo)
    if split_mode == 2:
        all_points = np.empty([0, num_points_upper_threshold, dataset_velo[0].shape[1]], dtype=np.float32)
        all_centroids = np.empty([0, dataset_velo[0].shape[1]], dtype=np.float32)
    else:
        all_points = np.empty([0, num_points_upper_threshold, dataset_velo[0].shape[1]-1], dtype=np.float32)
        all_centroids = np.empty([0, dataset_velo[0].shape[1]-1], dtype=np.float32)
    all_meta = np.empty([0], dtype=np.int32)

    print ('------ loading pointnet data ------')
    if split_mode == 3: # very large memory consumption when use multi-process
        for i in range(len(dataset_velo)):
            print ("finished clustering on batch %d" % i)
            cellmap = get_cellmap_from_raw_points(dataset_velo[i], n_threads=n_threads, split_mode=split_mode, cluster_mode=cluster_mode)
            batch_points, batch_meta = get_batch_data_from_cellmap(cellmap, num_points_upper_threshold=num_points_upper_threshold)
            batch_points = rescale(batch_points, batch_meta['number_points'], num_points_upper_threshold)
            all_points = np.concatenate([all_points, batch_points], axis=0)
            all_meta = np.concatenate([all_meta, batch_meta['number_points']], axis=0)
            all_centroids = np.concatenate([all_centroids, batch_meta['centroids']], axis=0)
    else:
        pool = Pool(n_threads)
        for i, cellmap in enumerate(pool.imap(partial(get_cellmap_from_raw_points, split_mode=split_mode), dataset_velo)):
            batch_points, batch_meta = get_batch_data_from_cellmap(cellmap, num_points_upper_threshold=num_points_upper_threshold)
            # batch_points = rescale(batch_points, batch_meta['number_points'], num_points_upper_threshold)
            all_points = np.concatenate([all_points, batch_points], axis=0)
            all_meta = np.concatenate([all_meta, batch_meta['number_points']], axis=0)
            all_centroids = np.concatenate([all_centroids, batch_meta['centroids']], axis=0)

        pool.close()
        pool.join()

    if shuffle == True:
        all_points, all_meta = shuffle_data(all_points, all_meta)

    all_points = all_points[:,:,:3]
    # all_points, all_mean = rescale(all_points, all_meta, num_points_upper_threshold=num_points_upper_threshold)

    return all_points, all_meta, all_centroids

def load_pc_octree_data(batch_size, train=True, split_mode=0, n_threads=16, octree_depth=3, shuffle=True):
    date = '2011_09_26'
    if train == True:
        drive = '0001'
    else:
        drive = '0005'
    basedir = '/scratch2/sniu/kitti'

    num_points_upper_threshold = 200

    dataset = load_dataset(date, drive, basedir)

    dataset_velo = list(dataset.velo)
    all_points = np.empty([0, num_points_upper_threshold, dataset_velo[0].shape[1]-1], dtype=np.float32)
    all_centroids = np.empty([0, dataset_velo[0].shape[1]-1], dtype=np.float32)
    all_meta = np.empty([0], dtype=np.int32)

    for i in range(len(dataset_velo)):
        cellmap = get_cellmap_from_raw_points(dataset_velo[i], n_threads=n_threads, split_mode=0)
        batch_points, batch_meta = get_batch_data_from_cellmap(cellmap, num_points_upper_threshold=num_points_upper_threshold)

        all_points = np.concatenate([all_points, batch_points], axis=0)
        all_meta = np.concatenate([all_meta, batch_meta['number_points']], axis=0)
        all_centroids = np.concatenate([all_centroids, batch_meta['centroids']], axis=0)

    return all_points, all_meta, all_centroids


def load_orig_pc_data(batch_size, train=True, n_threads=16, shuffle=True):
    date = '2011_09_26'
    if train == True:
        drive = '0001'
    else:
        drive = '0005'
    basedir = '/scratch2/sniu/kitti'

    dataset = load_dataset(date, drive, basedir)
    # for visualization
    tracklet_rects, tracklet_types = load_tracklets_for_frames(len(list(dataset.velo)), '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(basedir, date, date, drive))
    frame = 10 # for visualization
    display_frame_statistics(dataset, tracklet_rects, tracklet_types, frame, points=1.0) # original plot

    dataset_gray = list(dataset.gray)
    dataset_rgb = list(dataset.rgb)
    all_points = list(dataset.velo)

    return all_points, dataset_gray, dataset_rgb, tracklet_rects, tracklet_types


def reconstruct_points_from_batch_data(batch_data, meta_data):
    all_points = None
    batch_size = batch_data.shape[0]
    for i_batch in range(batch_size):

        real_number_points = meta_data['number_points'][i_batch]
        batch_points = batch_data[i_batch, :real_number_points, :]

        points = batch_points + meta_data['centroids'][i_batch]
        if all_points is None:
            all_points = points
        else:
            all_points = np.concatenate([all_points, points])
        
    return all_points

def visualize_3d_points(points, num=1000000, file_name='sample'):
    fig = plt.figure()
    num = min(points.shape[0], num)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:num, 0], points[:num, 1], points[:num, 2], c='r', marker='.')
    plt.axis('equal')
    plt.savefig(file_name)

def visualize_recon_points(points, recon, num=1000000, file_name='sample'):
    fig = plt.figure()
    num = min(points.shape[0], num)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:num, 0], points[:num, 1], points[:num, 2], c='r', marker='.')
    ax.scatter(recon[:num, 0], recon[:num, 1], recon[:num, 2], c='b', marker='.')
    plt.axis('equal')
    plt.savefig(file_name)


# def get_cellmap_from_raw_points(raw_points, cell_per_meter=1, cellmap_radius=30):

#     cellmap_size = 2*cell_per_meter*cellmap_radius

#     # Extract foreground points within cellmap
#     index_within_cellmap = (abs(raw_points[:, 0]) < cellmap_radius) &  (abs(raw_points[:, 1]) < cellmap_radius);
#     points_within_cellmap = raw_points[index_within_cellmap, :]
#     cell_id_2d = cellmap_radius*cell_per_meter + np.floor(points_within_cellmap[:, 0:2]*cell_per_meter)

#     row = cell_id_2d[:,0]
#     column = cell_id_2d[:,1]
#     cellmap_matrix = coo_matrix((np.ones_like(row), (row, column)), shape=(cellmap_size, cellmap_size)).toarray()

#     cell_id = sub2ind([cellmap_size, cellmap_size], cell_id_2d[:,0], cell_id_2d[:,1])
#     cell_id_unique = np.unique(cell_id) 

#     cellmap = {}
#     for i_cell in cell_id_unique:
#         cell_id_index = (cell_id == i_cell)
#         cellmap[i_cell] = points_within_cellmap[cell_id_index, :]

#     return cellmap, cell_id_2d, cell_id, points_within_cellmap, cellmap_matrix

def find_connected_components(cellmap_matrix, blur_radius = 0.5, threshold = 0.1):
    img = (cellmap_matrix!=0).astype(float) 
    imgf = ndimage.gaussian_filter(img, blur_radius)
    labeled, nr_objects = ndimage.label(imgf > threshold) 
    components = labeled * (cellmap_matrix!=0)
    return components, nr_objects

def find_component_points(components, nr_objects):

    component_points = []
    for i_object in range(1,nr_objects+1):
        cell_ids = np.where(components == i_object)
        cell_nums = cell_ids[0].shape[0]
        rows = []
        columns = []
        for i_cell in range(cell_nums):
            rows.append(cell_ids[0][i_cell])
            columns.append(cell_ids[1][i_cell])

        rows = np.stack(rows)
        columns = np.stack(columns)
        cell_ids_1D = sub2ind([cellmap_size, cellmap_size], rows, columns)

        points = None
        for cell_id_1D in cell_ids_1D:
            if points is None:
                points = cellmap[cell_id_1D]
            else:
                points = np.concatenate((points, cellmap[cell_id_1D]), axis=0)

        component_points.append(points)

    return component_points

