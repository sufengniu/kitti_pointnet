import numpy as np
import tensorflow as tf
import sys
import os
import os.path as osp
import pandas as pd
from tqdm import *
from functools import partial
from itertools import repeat

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from multiprocessing import Pool
import collections
import random
from random import shuffle, randint
import scipy
from scipy import ndimage
from scipy.sparse import coo_matrix

from source.utilities import print_progress
from moviepy.editor import ImageSequenceClip

from sklearn import linear_model, datasets
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn import mixture

from source import parseTrackletXML as xmlParser
import pykitti

from octree import octree_func

PointCell = collections.namedtuple('PointCell', (
    'data_id', 'cell_id', 'points', 'centroids', 'scale', 
    'num_points', 'mse', 'var', 'mean'))

colors = {
    'Car': 'b',
    'Tram': 'r',
    'Cyclist': 'g',
    'Van': 'c',
    'Truck': 'm',
    'Pedestrian': 'y',
    'Sitter': 'k'
}
# visualization limits
axes_limits = [
    [-20, 80], # X axis range
    [-20, 20], # Y axis range
    [-3, 10]   # Z axis range
]
axes_str = ['X', 'Y', 'Z']

def draw_box(pyplot_axis, vertices, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.
    
    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=0.5)

def display_frame_statistics(dataset, tracklet_rects, tracklet_types, frame, points=0.2):
    """
    Displays statistics for a single frame. Draws camera data, 3D plot of the lidar point cloud data and point cloud
    projections to various planes.
    
    Parameters
    ----------
    dataset         : `raw` dataset.
    tracklet_rects  : Dictionary with tracklet bounding boxes coordinates.
    tracklet_types  : Dictionary with tracklet types.
    frame           : Absolute number of the frame.
    points          : Fraction of lidar points to use. Defaults to `0.2`, e.g. 20%.
    """
    dataset_gray = list(dataset.gray)
    dataset_rgb = list(dataset.rgb)
    dataset_velo = list(dataset.velo)
    
    print('Frame timestamp: ' + str(dataset.timestamps[frame]))
    # Draw camera data
    f, ax = plt.subplots(2, 2, figsize=(15, 5))
    ax[0, 0].imshow(dataset_gray[frame][0], cmap='gray')
    ax[0, 0].set_title('Left Gray Image (cam0)')
    ax[0, 1].imshow(dataset_gray[frame][1], cmap='gray')
    ax[0, 1].set_title('Right Gray Image (cam1)')
    ax[1, 0].imshow(dataset_rgb[frame][0])
    ax[1, 0].set_title('Left RGB Image (cam2)')
    ax[1, 1].imshow(dataset_rgb[frame][1])
    ax[1, 1].set_title('Right RGB Image (cam3)')
    plt.savefig('camera')

    points_step = int(1. / points)
    point_size = 0.01 * (1. / points)
    velo_range = range(0, dataset_velo[frame].shape[0], points_step)
    velo_frame = dataset_velo[frame][velo_range, :]      
    def draw_point_cloud(ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
        """
        Convenient method for drawing various point cloud projections as a part of frame statistics.
        """
        ax.scatter(*np.transpose(velo_frame[:, axes]), s=point_size, c=velo_frame[:, 3], cmap='gray')
        ax.set_title(title)
        ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
        ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
        if len(axes) > 2:
            ax.set_xlim3d(*axes_limits[axes[0]])
            ax.set_ylim3d(*axes_limits[axes[1]])
            ax.set_zlim3d(*axes_limits[axes[2]])
            ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
        else:
            ax.set_xlim(*axes_limits[axes[0]])
            ax.set_ylim(*axes_limits[axes[1]])
        # User specified limits
        if xlim3d!=None:
            ax.set_xlim3d(xlim3d)
        if ylim3d!=None:
            ax.set_ylim3d(ylim3d)
        if zlim3d!=None:
            ax.set_zlim3d(zlim3d)
            
        for t_rects, t_type in zip(tracklet_rects[frame], tracklet_types[frame]):
            draw_box(ax, t_rects, axes=axes, color=colors[t_type])
            
    # Draw point cloud data as 3D plot
    f2 = plt.figure(figsize=(15, 8))
    ax2 = f2.add_subplot(111, projection='3d')                    
    draw_point_cloud(ax2, 'Velodyne scan', xlim3d=(-10,30))
    plt.savefig('3d_point_cloud')
    
    # Draw point cloud data as plane projections
    f, ax3 = plt.subplots(3, 1, figsize=(15, 25))
    draw_point_cloud(
        ax3[0], 
        'Velodyne scan, XZ projection (Y = 0), the car is moving in direction left to right', 
        axes=[0, 2] # X and Z axes
    )
    draw_point_cloud(
        ax3[1], 
        'Velodyne scan, XY projection (Z = 0), the car is moving in direction left to right', 
        axes=[0, 1] # X and Y axes
    )
    draw_point_cloud(
        ax3[2], 
        'Velodyne scan, YZ projection (X = 0), the car is moving towards the graph plane', 
        axes=[1, 2] # Y and Z axes
    )
    plt.savefig('2d_point_cloud')


def load_dataset(date, drive, basedir, calibrated=False, frame_range=None):
    """
    Loads the dataset with `date` and `drive`.
    
    Parameters
    ----------
    date        : Dataset creation date.
    drive       : Dataset drive.
    calibrated  : Flag indicating if we need to parse calibration data. Defaults to `False`.
    frame_range : Range of frames. Defaults to `None`.

    Returns
    -------
    Loaded dataset of type `raw`.
    """
    dataset = pykitti.raw(basedir, date, drive)

    # Load the data
    if calibrated:
        dataset.load_calib()  # Calibration data are accessible as named tuples

    np.set_printoptions(precision=4, suppress=True)
    print('\nDrive: ' + str(dataset.drive))
    print('\nFrame range: ' + str(dataset.frames))

    if calibrated:
        print('\nIMU-to-Velodyne transformation:\n' + str(dataset.calib.T_velo_imu))
        print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
        print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

    return dataset

def load_tracklets_for_frames(n_frames, xml_path):
    """
    Loads dataset labels also referred to as tracklets, saving them individually for each frame.

    Parameters
    ----------
    n_frames    : Number of frames in the dataset.
    xml_path    : Path to the tracklets XML.

    Returns
    -------
    Tuple of dictionaries with integer keys corresponding to absolute frame numbers and arrays as values. First array
    contains coordinates of bounding box vertices for each object in the frame, and the second array contains objects
    types as strings.
    """
    tracklets = xmlParser.parseXML(xml_path)

    frame_tracklets = {}
    frame_tracklets_types = {}
    for i in range(n_frames):
        frame_tracklets[i] = []
        frame_tracklets_types[i] = []

    # loop over tracklets
    for i, tracklet in enumerate(tracklets):
        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h, w, l = tracklet.size
        # in velodyne coordinates around zero point and without orientation yet
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])
        # loop over all data in tracklet
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
            # determine if object is in the image; otherwise continue
            if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                continue
            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]  # other rotations are supposedly 0
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]
            ])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
            frame_tracklets[absoluteFrameNumber] = frame_tracklets[absoluteFrameNumber] + [cornerPosInVelo]
            frame_tracklets_types[absoluteFrameNumber] = frame_tracklets_types[absoluteFrameNumber] + [
                tracklet.objectType]

    return (frame_tracklets, frame_tracklets_types)

def draw_3d_plot(frame, dataset, tracklet_rects, tracklet_types, points=0.2):
    """
    Saves a single frame for an animation: a 3D plot of the lidar data without ticks and all frame trackelts.
    Parameters
    ----------
    frame           : Absolute number of the frame.
    dataset         : `raw` dataset.
    tracklet_rects  : Dictionary with tracklet bounding boxes coordinates.
    tracklet_types  : Dictionary with tracklet types.
    points          : Fraction of lidar points to use. Defaults to `0.2`, e.g. 20%.

    Returns
    -------
    Saved frame filename.
    """
    dataset_velo = list(dataset.velo)
    
    f = plt.figure(figsize=(12, 8))
    axis = f.add_subplot(111, projection='3d', xticks=[], yticks=[], zticks=[])

    points_step = int(1. / points)
    point_size = 0.01 * (1. / points)
    velo_range = range(0, dataset_velo[frame].shape[0], points_step)
    velo_frame = dataset_velo[frame][velo_range, :]
    axis.scatter(*np.transpose(velo_frame[:, [0, 1, 2]]), s=point_size, c=velo_frame[:, 3], cmap='gray')
    axis.set_xlim3d(*axes_limits[0])
    axis.set_ylim3d(*axes_limits[1])
    axis.set_zlim3d(*axes_limits[2])
    for t_rects, t_type in zip(tracklet_rects[frame], tracklet_types[frame]):
        draw_box(axis, t_rects, axes=[0, 1, 2], color=colors[t_type])
    filename = 'video/frame_{0:0>4}.png'.format(frame)
    plt.savefig(filename)
    plt.close(f)
    return filename

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

def get_cluster_cellmap_from_raw_points(raw_points, n_threads=16, cluster_mode="kmeans", seeds=[0]):

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

def get_cellmap_from_raw_points(raw_points, is_training=False, 
                    n_threads=16, split_mode=0, cluster_mode='kmeans', 
                    cell_per_meter=1, cellmap_radius=30):
    
    if is_training == True:
        seeds = [0,22123,3010,4165]
    else:
        seeds = [0]

    foreground_points, background_points = rough_LBS(raw_points)
    if split_mode == 0:
        point = foreground_points
    elif split_mode == 1:
        point = background_points
    elif split_mode == 2:
        point = raw_points[:,:2]
    elif split_mode == 3:
        return get_cluster_cellmap_from_raw_points(foreground_points, n_threads, cluster_mode=cluster_mode, seeds=seeds)
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
    cell_id = []

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

        cell_id.append(i_cell)

        # data same shape
        if num_points < num_points_upper_threshold:
            cellpoints = np.concatenate([points, np.zeros((num_points_upper_threshold-num_points, dim))],axis=0)
        else:
            cellpoints = points

        batch_data.append(cellpoints)

    # batch * num_points_upper_threshold * 4
    batch_data = np.stack(batch_data)

    meta_data = {}
    meta_data['number_points'] = np.stack(number_points) # batch
    meta_data['centroids'] = np.stack(centroids) # batch
    meta_data['cell_id'] = np.stack(cell_id) # batch
    return batch_data, meta_data

def shuffle_data(cell_id, point_clouds, meta, scale, centroids, seed=12345):
    num_examples = point_clouds.shape[0]
    if seed is not None:
        np.random.seed(seed)
    perm = np.arange(num_examples)
    np.random.shuffle(perm)
    return cell_id[perm], point_clouds[perm], meta[perm], scale[perm], centroids[perm]

def rescale(point_cloud, num_points):
    p_scaled = point_cloud[:num_points, :]
    new_point = np.zeros_like(point_cloud)
    center = np.mean(p_scaled, 0)
    p_scaled -= center
    x_max, y_max, z_max = np.max(p_scaled, 0)
    x_min, y_min, z_min = np.min(p_scaled, 0)
    x_var, y_var, z_var = max(x_max,abs(x_min)), max(y_max,abs(y_min)), max(z_max,abs(z_min))
    ratio = 1. / max(x_var, y_var, z_var)
    new_point[:num_points] = p_scaled * ratio
    return new_point, ratio, center

class LoadData():
    def __init__(self, args, drives):
        
        self.batch_size = args.batch_size
        self.batch_idx = 0
        self.basedir = args.data_in_dir
        self.L = list(map(int, args.PL.split(',')))
        self.W = list(map(int, args.PL.split(',')))
        self.factor = self.L[1] / self.L[0] if len(self.L) > 1 else 0
        self.level = len(self.L)
        self.fb_split = args.fb_split
        self.cell_max_points = list(map(int, args.cell_max_points.split(',')))
        self.cell_min_points = list(map(int, args.cell_min_points.split(',')))
        self.max_x_dist = 16.0
        self.max_y_dist = 16.0
        self.max_z_dist = 3.0

        self.columns = ['points', 'parent', 'index', 'ratio', 'center', 'level', 'num_points']
        if self.level > 1:
            self.cell_set = []
            for i in range(self.level):
                self.cell_set.append(pd.DataFrame(columns=self.columns))
        else:
            self.cell_set = [pd.DataFrame(columns=self.columns)]

        self.data_id = []
        self.dataset_gray, self.dataset_rgb, self.dataset_velo = [], [], []
        for drive in drives:
            dataset = load_dataset(args.date, drive, self.basedir)
            self.data_id += [args.date + '_' + drive]
            # for visualization
            # tracklet_rects, tracklet_types = load_tracklets_for_frames(len(list(dataset.velo)), '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(basedir, date, date, drive))
            # frame = 10 # for visualization
            # display_frame_statistics(dataset, tracklet_rects, tracklet_types, frame)
            self.dataset_gray += list(dataset.gray)
            self.dataset_rgb += list(dataset.rgb)
            self.dataset_velo += list(dataset.velo)
        self.cleaned_velo = []
        for points in self.dataset_velo:
            idx_z = np.squeeze(np.argwhere(abs(points[:,2]) > self.max_z_dist)) # remove all points below -3.0 in z-axis
            idx_x = np.squeeze(np.argwhere(abs(points[:,0]) > self.max_x_dist)) # remove all points above 32m
            idx_y = np.squeeze(np.argwhere(abs(points[:,1]) > self.max_y_dist)) # remove all points above 32m
            idx = np.union1d(np.union1d(idx_x, idx_y), idx_z)
            filter_data = np.delete(points, idx, axis=0)[:,:3]
            # testing
            # index = random.sample(range(0, len(filter_data)-1), 20000)
            # self.cleaned_velo.append(filter_data[index])
            self.cleaned_velo.append(filter_data) # ignore the 4th dimension

        test_sweep_split = int(0.95*len(self.cleaned_velo))
        self.test_sweep = self.cleaned_velo[test_sweep_split:]

        if self.fb_split == True:
            self.fb_partition_batch(self.cleaned_velo[:test_sweep_split])

            print ("total foreground training set: {}, validation set: {}, testing set: {}".format(self.train_cell_f[0].shape[0], 
                                                                                                   self.valid_cell_f[0].shape[0],
                                                                                                   self.test_cell_f[0].shape[0]))  
            print ("total background training set: {}, validation set: {}, testing set: {}".format(self.train_cell_b[0].shape[0], 
                                                                                                   self.valid_cell_b[0].shape[0],
                                                                                                   self.test_cell_b[0].shape[0]))

        else:
            self.train_cell, \
            self.valid_cell, \
            self.test_cell = self.partition_batch(self.cleaned_velo[:test_sweep_split])

            print ("total training set: {}, validation set: {}, testing set: {}".format(self.train_cell[0].shape[0], 
                                                                                        self.valid_cell[0].shape[0],
                                                                                        self.test_cell[0].shape[0]))
        print ("batch size: {}".format(self.batch_size))

        # sample
        self.sample_idx = randint(0, len(self.test_sweep)-1)
        self.sample_idx_train = randint(0, len(self.cleaned_velo[:test_sweep_split])-1)

        if self.fb_split == True:
            f_points, b_points = rough_LBS(self.test_sweep[self.sample_idx])
            self.test_f_sweep, self.test_b_sweep = f_points, b_points
            self.sample_points_f = self.partition_single(f_points, self.sample_idx)
            self.sample_points_b = self.partition_single(b_points, self.sample_idx)
            # training set, see if overfitting
            f_points_t, b_points_t = rough_LBS(self.cleaned_velo[self.sample_idx_train])
            self.train_f_sweep, self.train_b_sweep = f_points_t, b_points_t
            self.train_points_f = self.partition_single(f_points_t, self.sample_idx_train)
            self.train_points_b = self.partition_single(b_points_t, self.sample_idx_train)
            
        else:
            self.sample_points = self.partition_single(self.test_sweep[self.sample_idx], self.sample_idx)
            # training set, see if overfitting
            self.train_points = self.partition_single(self.cleaned_velo[self.sample_idx_train], self.sample_idx_train)

    def fb_partition_batch(self, cleaned_velo):

        foreground, background = [], []
        for points in cleaned_velo:
            f_points, b_points = rough_LBS(points)
            foreground.append(f_points)
            background.append(b_points)

        train_cell_f, \
        valid_cell_f, \
        test_cell_f = self.partition_batch(foreground)

        train_cell_b, \
        valid_cell_b, \
        test_cell_b = self.partition_batch(background)

        self.train_cell_f, self.valid_cell_f, self.test_cell_f = train_cell_f, valid_cell_f, test_cell_f
        self.train_cell_b, self.valid_cell_b, self.test_cell_b = train_cell_b, valid_cell_b, test_cell_b
        return

    def partition_batch(self, cleaned_velo):
        cell_points = [[] for _ in range(self.level)]
        print ('partitioning...')
        for idx, points in enumerate(tqdm(cleaned_velo)):
            for l in range(self.level):
                cell_points_buf = self.partition_single(points, idx)
                cell_points[l].append(cell_points_buf[l])

        train_cell, valid_cell, test_cell = [], [], []
        for l in range(self.level):
            cell_points[l] = pd.concat(cell_points[l])
            self.cell_set[l] = cell_points[l]
            train_split = int(0.9 * cell_points[l].shape[0])
            valid_split = int(0.95 * cell_points[l].shape[0])

            train_buf = cell_points[l][:train_split]
            train_buf = train_buf.iloc[np.random.permutation(np.arange(len(train_buf)))]
            train_cell.append(train_buf)
            valid_cell.append(cell_points[l][train_split:valid_split])
            test_cell.append(cell_points[l][valid_split:])

        return train_cell, valid_cell, test_cell
        
    def partition_single(self, points, idx):
        xyz_max = np.max(points, 0)
        xyz_min = np.min(points, 0)
        
        multi_cell = []
        # partition into a LxW grid
        for k in range(self.level):
            L, W = self.L[k], self.W[k]
            cell_width, cell_length = (xyz_max[0]-xyz_min[0])/W, (xyz_max[1]-xyz_min[1])/L
            index = np.arange(L*W)
            cell_points = pd.DataFrame(index=index, columns=self.columns)
            for i in range(L):
                for j in range(W):
                    left, right = xyz_min[0]+i*cell_width, xyz_min[0]+(i+1)*cell_width 
                    up, down = xyz_min[1]+j*cell_length, xyz_min[1]+(j+1)*cell_length
                    cell_idx = np.where(np.logical_and(np.logical_and(points[:,0]>=left, points[:,0]<right), np.logical_and(points[:,1]>=up, points[:,1]<down)))[0]
                    num_points = len(cell_idx)

                    if num_points < self.cell_min_points[k]:
                        continue

                    if num_points > self.cell_max_points[k]: # if greater than max points, random sample
                        sample_index = np.random.choice(num_points, self.cell_max_points[k], replace=False)
                        cell, ratio, center = rescale(points[cell_idx][sample_index], self.cell_max_points[k])
                    elif num_points < self.cell_max_points[k]:
                        new_point = np.zeros([self.cell_max_points[k], 3])
                        new_point[:num_points] = points[cell_idx]
                        cell, ratio, center = rescale(new_point, num_points)

                    cell_idx = i*W+j
                    cell_points.iloc[cell_idx]['points'] = cell
                    if k > 0:
                        I, J = i//self.factor, j//self.factor
                        parent_idx = int(I*self.W[k-1]+J)
                        cell_points.iloc[cell_idx]['parent'] = multi_cell[-1].iloc[parent_idx]['points']
                    cell_points.iloc[cell_idx]['index'] = idx
                    cell_points.iloc[cell_idx]['ratio'] = ratio
                    cell_points.iloc[cell_idx]['center'] = center
                    cell_points.iloc[cell_idx]['level'] = k
                    cell_points.iloc[cell_idx]['num_points'] = min(num_points, self.cell_max_points[k])
            multi_cell.append(cell_points)

        for k in range(self.level):
            multi_cell[k] = multi_cell[k].dropna(thresh=2)
        return multi_cell

    def reconstruct_scene(self, cell_points, cell_info):
        scene = []
        for i in range(len(cell_info)):
            cell = cell_points[i]
            num_points = cell_info.iloc[i]['num_points']
            scaled_cell = cell[:num_points] / cell_info.iloc[i]['ratio']
            scaled_cell += cell_info.iloc[i]['center']
            scene.append(scaled_cell)
        return np.concatenate(scene, axis=0)

    def all_data(self):
        return self.cell_points

    
def load_pc_octree_data(batch_size, data_in_dir, date, drive, num_points_upper_threshold,
                        split_mode=0, n_threads=16, octree_depth=3, shuffle=True):
    basedir = data_in_dir
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

def plot_3d_points(points, num=1000000, file_name='sample'):
    fig = plt.figure()
    num = min(points.shape[0], num)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:num, 0], points[:num, 1], points[:num, 2], c='b', marker='.')
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

def visualize_3d_points(dataset, points=1.0, dir='.', filename='sample'):
    """
    Displays statistics for a single frame. 3D plot of the lidar point cloud data and point cloud
    projections to various planes.
    
    Parameters
    ----------
    dataset         : `raw` dataset.
    points          : Fraction of lidar points to use. Defaults to `0.2`, e.g. 20%.
    """

    dataset_velo = dataset
    
    points_step = int(1. / points)
    point_size = 0.05 * (1. / points)
    velo_range = range(0, dataset_velo.shape[0], points_step)
    velo_frame = dataset_velo[velo_range, :]      
    def draw_point_cloud(ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
        """
        Convenient method for drawing various point cloud projections as a part of frame statistics.
        """
        ax.scatter(*np.transpose(velo_frame[:, axes]), s=point_size, cmap='gray')
        ax.set_title(title)
        ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
        ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
        if len(axes) > 2:
            ax.set_xlim3d(*axes_limits[axes[0]])
            ax.set_ylim3d(*axes_limits[axes[1]])
            ax.set_zlim3d(*axes_limits[axes[2]])
            ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
        else:
            # ax.set_xlim(-30, 30)
            # ax.set_ylim(-30, 30)
            ax.set_xlim(*axes_limits[axes[0]])
            ax.set_ylim(*axes_limits[axes[1]])
        # User specified limits
        if xlim3d!=None:
            ax.set_xlim3d(xlim3d)
        if ylim3d!=None:
            ax.set_ylim3d(ylim3d)
        if zlim3d!=None:
            ax.set_zlim3d(zlim3d)
                        
    # Draw point cloud data as 3D plot
    f2 = plt.figure(figsize=(15, 8))
    ax2 = f2.add_subplot(111, projection='3d')                    
    draw_point_cloud(ax2, 'Velodyne scan', xlim3d=(-10,30))
    plt.savefig(dir+'/3d_point_cloud_'+filename)
    
    # Draw point cloud data as plane projections
    f, ax3 = plt.subplots(3, 1, figsize=(15, 25))
    draw_point_cloud(
        ax3[0], 
        'Velodyne scan, XZ projection (Y = 0), the car is moving in direction left to right', 
        axes=[0, 2] # X and Z axes
    )
    draw_point_cloud(
        ax3[1], 
        'Velodyne scan, XY projection (Z = 0), the car is moving in direction left to right', 
        axes=[0, 1] # X and Y axes
    )
    draw_point_cloud(
        ax3[2], 
        'Velodyne scan, YZ projection (X = 0), the car is moving towards the graph plane', 
        axes=[1, 2] # Y and Z axes
    )
    plt.savefig(dir+'/2d_point_cloud_'+filename)



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



def dist_stat(recon, orig, meta, cell_max_points=200):
    batch_size = recon.shape[0]
    num_anchor = recon.shape[1]
    num_points = orig.shape[1]

    all_dist = np.empty([0])
    if num_anchor == num_points:
        for i in range(batch_size):
            idx = np.arange(min(meta[i], cell_max_points))
            dist_tmp = scipy.spatial.distance.cdist(orig[i, idx], recon[i, idx])
            dist_vec = np.min(dist_tmp, 1) # [num_points]
            all_dist = np.concatenate([all_dist, dist_vec])
    elif num_anchor < num_points:
        for i in range(batch_size):
            idx = np.arange(min(meta[i], num_anchor))
            idx_orig = np.arange(min(meta[i], cell_max_points))
            dist_tmp = scipy.spatial.distance.cdist(orig[i, idx_orig], recon[i, idx])
            dist_vec = np.min(dist_tmp, 1) # [num_points]
            all_dist = np.concatenate([all_dist, dist_vec])

    return np.mean(all_dist), np.var(all_dist), np.mean(all_dist**2)

def sweep_stat(recon, orig):
    dist = scipy.spatial.distance.cdist(recon, orig) # order must be recon, orig
    dist_vec = np.min(dist, 1)
    return np.mean(dist_vec), np.var(dist_vec), np.mean(dist_vec**2)

def dist_vis(recon, orig, meta, num_points_upper_threshold=200):
    batch_size = recon.shape[0]
    num_anchor = recon.shape[1]
    num_points = orig.shape[1]

    all_dist = np.empty([0])
    if num_anchor == num_points:
        for i in range(batch_size):
            idx = np.arange(min(meta[i], num_points_upper_threshold))
            dist_tmp = scipy.spatial.distance.cdist(orig[i, idx], recon[i, idx])
            dist_vec = np.min(dist_tmp, 1) # [num_points]
            all_dist = np.concatenate([all_dist, dist_vec])
    elif num_anchor < num_points:
        for i in range(batch_size):
            idx = np.arange(min(meta[i], num_anchor))
            idx_orig = np.arange(min(meta[i], num_points_upper_threshold))
            dist_tmp = scipy.spatial.distance.cdist(orig[i, idx_orig], recon[i, idx])
            dist_vec = np.min(dist_tmp, 1) # [num_points]
            all_dist = np.concatenate([all_dist, dist_vec])

    pd.DataFrame(all_dist).plot(kind='density')
    plt.savefig('dist_distribution')
    plt.close()

def dist_octree_stat(recon, orig, meta, num_points_upper_threshold=200):
    batch_size = len(recon)
    num_anchor = np.array([recon[i].shape[0] for i in range(batch_size)])
    num_points = orig.shape[1]

    all_dist = np.empty([0])
    for i in range(batch_size):
        if recon[i].shape[0] == 0:
            continue
        idx = np.arange(min(meta[i], num_anchor[i]))
        idx_orig = np.arange(min(meta[i], num_points_upper_threshold))
        dist_tmp = scipy.spatial.distance.cdist(orig[i, idx_orig], recon[i][idx])
        dist_vec = np.min(dist_tmp, 1) # [num_points]
        all_dist = np.concatenate([all_dist, dist_vec])

    return np.mean(all_dist), np.var(all_dist), np.sum(all_dist**2)

def create_dir(dir_path):
    ''' Creates a directory (or nested directories) if they don't exist.
    '''
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path




        # all_points = np.empty([0, num_points_upper_threshold, self.dataset_velo[0].shape[1]-1], dtype=np.float32)
        # all_centroids = np.empty([0, self.dataset_velo[0].shape[1]-1], dtype=np.float32)
        # all_meta = np.empty([0], dtype=np.int32)
        # all_cell_id = np.empty([0, 2], dtype=np.int32)
        # all_scale = np.empty([0], dtype=np.float32)

        # print ('------ loading pointnet data ------')
        # if partition_mode == 4: # very large memory consumption when use multi-process
        #     for i in range(len(self.dataset_velo)):
        #         print ("=> start clustering on batch %d" % i)
        #         cellmap = get_cellmap_from_raw_points(self.dataset_velo[i], n_threads=n_threads, 
        #                                 split_mode=partition_mode, cluster_mode=cluster_mode, is_training=is_training)
        #         batch_points, batch_meta = get_batch_data_from_cellmap(cellmap, num_points_upper_threshold=num_points_upper_threshold)
        #         batch_points = rescale(batch_points, batch_meta['number_points'], num_points_upper_threshold)

        #         all_cell_id = np.concatenate([all_cell_id, np.stack([i*np.ones_like(batch_meta['cell_id']),
        #                                       batch_meta['cell_id']]).transpose()], axis=0)
        #         all_points = np.concatenate([all_points, batch_points], axis=0)
        #         all_meta = np.concatenate([all_meta, batch_meta['number_points']], axis=0)
        #         all_centroids = np.concatenate([all_centroids, batch_meta['centroids']], axis=0)

        # else:
        #     pool = Pool(n_threads)
        #     for i, cellmap in enumerate(pool.imap(partial(get_cellmap_from_raw_points, split_mode=partition_mode), self.dataset_velo)):
        #         batch_points, batch_meta = get_batch_data_from_cellmap(cellmap, num_points_upper_threshold=num_points_upper_threshold)

        #         all_cell_id = np.concatenate([all_cell_id, np.stack([i*np.ones_like(batch_meta['cell_id']), 
        #                                       batch_meta['cell_id']]).transpose()], axis=0)
        #         all_points = np.concatenate([all_points, batch_points], axis=0)
        #         all_meta = np.concatenate([all_meta, batch_meta['number_points']], axis=0)
        #         all_scale = np.concatenate([all_meta, np.ones_like(batch_meta['number_points']).astype(np.float32)], axis=0)
        #         all_centroids = np.concatenate([all_centroids, batch_meta['centroids']], axis=0)

        #     pool.close()
        #     pool.join()

        # self.all_cell_id = all_cell_id
        # self.all_points = all_points
        # self.all_meta = all_meta
        # self.all_scale = all_scale
        # self.all_centroids = all_centroids

        # self.point_cell = []
        # for i in range(all_points.shape[0]):
        #     self.point_cell.append(PointCell(
        #                 data_id=self.data_id,
        #                 cell_id=all_cell_id[i],
        #                 points=all_points[i],
        #                 num_points=all_meta[i],
        #                 scale=all_scale[i],
        #                 centroids=all_centroids[i],
        #                 mse=-1.0,
        #                 var=-1.0,
        #                 mean=-1.0))

        # self.reset()

    # def next_batch(self):
    #     start_idx = self.batch_idx * self.batch_size
    #     end_idx = min((self.batch_idx+1) * self.batch_size, len(self.point_cell))
    #     batch_data = PointCell(
    #                 data_id=self.data_id,
    #                 cell_id=self.all_cell_id[start_idx:end_idx],
    #                 points=self.all_points[start_idx:end_idx],
    #                 num_points=self.all_meta[start_idx:end_idx],
    #                 scale=self.all_scale[start_idx:end_idx],
    #                 centroids=self.all_centroids[start_idx:end_idx],
    #                 mse=-1.0,
    #                 var=-1.0,
    #                 mean=-1.0)
    #     if end_idx == len(self.point_cell):
    #         self.reset()
    #     else:
    #         self.batch_idx += 1
    #     return batch_data
