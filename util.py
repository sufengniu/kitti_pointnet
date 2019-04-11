import numpy as np
import tensorflow as tf
import sys
import os
import os.path as osp
import pandas as pd
from tqdm import *
from functools import partial
from itertools import repeat, product
import h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import pickle
from multiprocessing import Pool
import collections
import random
from random import shuffle, randint
import scipy
from scipy import ndimage
from scipy.sparse import coo_matrix

from source.utilities import print_progress
# from moviepy.editor import ImageSequenceClip

from sklearn import linear_model, datasets
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn import mixture
from sklearn.neighbors import NearestNeighbors

from external.structural_losses.tf_nndistance import nn_distance
from external.structural_losses.tf_approxmatch import approx_match, match_cost

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
# axes_limits = [
#     [-20, 80], # X axis range
#     [-20, 20], # Y axis range
#     [-3, 10]   # Z axis range
# ]
# hdmap visualization limits
axes_limits = [
    [-32, 30], # X axis range
    [-100, 100], # Y axis range
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
    if num_points == 1:
        return np.zeros_like(point_cloud), 1.0, point_cloud[0]

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

def plot_distributions(df, labels, figsize, h, w):
    i=0
    fig = plt.figure(figsize=figsize)
    for l in labels:
        i+=1
        fig.add_subplot(h,w,i)
        plt.tight_layout()
        plt.title(f'{l}, mean = {df[l].mean():.2f}')
        lq = df[l].quantile(0.01)
        uq = df[l].quantile(0.99)
        feature = df[l].dropna()
        plt.hist(feature[(feature > lq)&(feature < uq)], density=True, bins=100, alpha=0.7)
    plt.show()

def corr_plot(df):
    corr = df.corr()

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

class LoadData():
    def __init__(self, args, train_drives, test_drives):
        
        self.batch_size = args.batch_size
        self.batch_idx = 0
        self.basedir = args.data_in_dir
        self.L = list(map(int, args.PL.split(',')))
        self.W = list(map(int, args.PW.split(',')))
        self.factor = self.L[1] / self.L[0] if len(self.L) > 1 else 1
        self.level = len(self.L)
        self.fb_split = args.fb_split
        self.cell_max_points = list(map(int, args.cell_max_points.split(',')))
        self.cell_min_points = list(map(int, args.cell_min_points.split(',')))
        self.max_x_dist = 16.0
        self.max_y_dist = 16.0
        self.max_z_dist = 3.0
        self.latent_dim = args.latent_dim
        self.range_view = True if args.partition_mode == 'range' else False
        self.num_sample_points = 5000
        self.load_all_kitti = (args.compress_all_kitti==True)

        self.num_hdmap_test = 1
        self.lower_min_elevate_angle = -24.33
        self.upper_min_elevate_angle = -8.33
        self.min_azimuth_angle = -180.0
        self.range_of_azimuth_angle = 360.0
        self.num_lasers = 64
        self.group_num_lasers = self.L # can tune
        self.image_height = [int(self.num_lasers / gnl) for gnl in self.group_num_lasers]
        self.image_width = self.W # original 64
        self.dataset = args.dataset
        
        self.columns = ['points', 'stacked', 'stacked_num', 'index', 'ratio',
                        'center', 'level', 'num_points', 'compressed', 'row', 'col']
        
        if self.level > 1:
            self.cell_set = []
            for i in range(self.level):
                self.cell_set.append(pd.DataFrame(columns=self.columns))
        else:
            self.cell_set = [pd.DataFrame(columns=self.columns)]

        # processing training drives
        self.data_id = []
        self.train_dataset_gray, self.train_dataset_rgb, self.train_dataset_velo = [], [], []
        self.test_dataset_gray, self.test_dataset_rgb, self.test_dataset_velo = [], [], []
        
        self.num_to_select = 12
        if self.dataset == 'kitti':
            self.load_kitti(args, train_drives, test_drives)
        elif self.dataset == 'hdmap':
            self.load_hdmap(args)
        else:
            print ("Error: No such dataset")
            raise

        self.test_sweep = self.test_cleaned_velo

        print ('partitioning testing set...')
        self.test_cell = self.partition_batch(self.test_cleaned_velo, permutation=False)
        print ("total testing set: {}".format(self.test_cell[0].shape[0]))

        # sample
        random.seed(10)
        self.sample_idx = randint(0, len(self.test_sweep)-1)
        self.sample_idx_train = randint(0, len(self.train_cleaned_velo)-1)
        self.sample_sweep = self.test_sweep[self.sample_idx]

        if self.fb_split == True:
            f_points, b_points = rough_LBS(self.test_sweep[self.sample_idx])
            self.test_f_sweep, self.test_b_sweep = f_points, b_points
            self.sample_points_f = self.partition_single(f_points, self.sample_idx)
            self.sample_points_b = self.partition_single(b_points, self.sample_idx)
            # training set, see if overfitting
            # f_points_t, b_points_t = rough_LBS(self.train_cleaned_velo[self.sample_idx_train])
            # self.train_f_sweep, self.train_b_sweep = f_points_t, b_points_t
            # self.train_points_f = self.partition_single(f_points_t, self.sample_idx_train)
            # self.train_points_b = self.partition_single(b_points_t, self.sample_idx_train)
            
        else:
            if self.range_view == True:
                self.sample_points = self.partition_single_range(self.test_sweep[self.sample_idx], self.sample_idx)
            else:
                self.sample_points = self.partition_single(self.test_sweep[self.sample_idx], self.sample_idx)
            # training set, see if overfitting
            # self.train_points = self.partition_single(self.train_cleaned_velo[self.sample_idx_train], self.sample_idx_train)

    def load_kitti(self, args, train_drives, test_drives):
        print ('loading data ...')
        if osp.isfile(osp.join(args.data_in_dir, 'training.h5')):
            self.train_cleaned_velo = []
            with h5py.File(self.basedir+'training.h5', 'r') as f:
                for k in f.keys():
                    self.train_cleaned_velo.append(f[k].value)
        else:
            for drive in train_drives:
                dataset = load_dataset(args.date, drive, self.basedir)
                self.data_id += [args.date + '_' + drive]
                data_len = len(list(dataset.velo))
                if self.load_all_kitti == False:
                    Idx = random.sample(range(data_len), self.num_to_select)
                    # for visualization
                    # tracklet_rects, tracklet_types = load_tracklets_for_frames(len(list(dataset.velo)), '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(basedir, date, date, drive))
                    # frame = 10 # for visualization
                    # display_frame_statistics(dataset, tracklet_rects, tracklet_types, frame)
                    self.train_dataset_gray += [list(dataset.gray)[i] for i in Idx]
                    self.train_dataset_rgb += [list(dataset.rgb)[i] for i in Idx]
                    self.train_dataset_velo += [list(dataset.velo)[i] for i in Idx]
                    # self.train_dataset_velo += list(dataset.velo)
                else:
                    # for visualization
                    # tracklet_rects, tracklet_types = load_tracklets_for_frames(len(list(dataset.velo)), '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(basedir, date, date, drive))
                    # frame = 10 # for visualization
                    # display_frame_statistics(dataset, tracklet_rects, tracklet_types, frame)
                    self.train_dataset_gray += list(dataset.gray)
                    self.train_dataset_rgb += list(dataset.rgb)
                    self.train_dataset_velo += list(dataset.velo)

            self.train_cleaned_velo = []
            for points in self.train_dataset_velo:
                idx_z = np.squeeze(np.argwhere(abs(points[:,2]) > self.max_z_dist)) # remove all points below -3.0 in z-axis
                idx_x = np.squeeze(np.argwhere(abs(points[:,0]) > self.max_x_dist)) # remove all points above 32m
                idx_y = np.squeeze(np.argwhere(abs(points[:,1]) > self.max_y_dist)) # remove all points above 32m
                idx = np.union1d(np.union1d(idx_x, idx_y), idx_z)
                filter_data = np.delete(points, idx, axis=0)[:,:3]
                # testing
                # index = random.sample(range(0, len(filter_data)-1), 20000)
                # self.cleaned_velo.append(filter_data[index])
                self.train_cleaned_velo.append(filter_data) # ignore the 4th dimension

            with h5py.File(self.basedir+'training.h5', 'w', libver='latest') as f:
                for idx, arr in enumerate(self.train_cleaned_velo):
                    dset = f.create_dataset(str(idx), data=arr)

        if osp.isfile(osp.join(args.data_in_dir, 'testing.h5')):
            self.test_cleaned_velo = []
            with h5py.File(self.basedir+'testing.h5', 'r') as f:
                for k in f.keys():
                    self.test_cleaned_velo.append(f[k].value)
        else:
            random.seed(0)
            for drive in test_drives:
                dataset = load_dataset(args.date, drive, self.basedir)
                self.data_id += [args.date + '_' + drive]
                data_len = len(list(dataset.velo))
                if self.load_all_kitti == False:
                    Idx = random.sample(range(data_len), self.num_to_select)
                    self.test_dataset_gray += [list(dataset.gray)[i] for i in Idx]
                    self.test_dataset_rgb += [list(dataset.rgb)[i] for i in Idx]
                    self.test_dataset_velo += [list(dataset.velo)[i] for i in Idx]
                else:
                    self.test_dataset_gray += list(dataset.gray)
                    self.test_dataset_rgb += list(dataset.rgb)
                    self.test_dataset_velo += list(dataset.velo)

            self.test_cleaned_velo = []
            for points in self.test_dataset_velo:
                idx_z = np.squeeze(np.argwhere(abs(points[:,2]) > self.max_z_dist)) # remove all points below -3.0 in z-axis
                idx_x = np.squeeze(np.argwhere(abs(points[:,0]) > self.max_x_dist)) # remove all points above 32m
                idx_y = np.squeeze(np.argwhere(abs(points[:,1]) > self.max_y_dist)) # remove all points above 32m
                idx = np.union1d(np.union1d(idx_x, idx_y), idx_z)
                filter_data = np.delete(points, idx, axis=0)[:,:3]
                self.test_cleaned_velo.append(filter_data) # ignore the 4th dimension

            with h5py.File(self.basedir+'testing.h5', 'w', libver='latest') as f:
                for idx, arr in enumerate(self.test_cleaned_velo):
                    dset = f.create_dataset(str(idx), data=arr)
        print ('loading data done, number of traininig set: %d, number of testing set %d'%(len(self.train_cleaned_velo), len(self.test_cleaned_velo)))

    def filter_hdmap(self):

        pc = None
        for cell in cells:
            if pc is None:
                pc = point_cloud_dict[cell]
            else:
                pc = np.concatenate([pc, point_cloud_dict[cell]], axis=0)

    def _get_mean(self, sweep):
        num = sweep.shape[0]
        center_x, center_y = 0, 0
        for i in range(num):
            center_x += sweep[i, 0]
            center_y += sweep[i, 1]
        center_x /= num
        center_y /= num
        return np.array([center_x, center_y, 0.0])

    def load_hdmap(self, args):
        f_list = []
        extensions = {'.pickle'}
        for subdir, dirs, files in os.walk(args.hdmap_dir):
            for file in files:
                ext = os.path.splitext(file)[-1].lower()
                if ext in extensions:
                    filename = os.path.join(subdir, file)
                    f_list.append(filename)

        train_f_list, test_f_list = f_list[:-self.num_hdmap_test], f_list[-self.num_hdmap_test:]
        self.train_cleaned_velo = []
        self.test_cleaned_velo = []

        for idx, file in enumerate(train_f_list):
            sweep = pickle.load(open(file, "rb"))
            for cell_pos, k in enumerate(sweep.keys()):
                center = self._get_mean(sweep[k])
                centered_sweep = sweep[k] - center
                self.train_cleaned_velo.append(centered_sweep)

        select_cells = [(10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11), (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 12), (12, 13), (12, 14), (12, 15), (12, 16), (12, 17), (12, 18), (13, 7), (13, 8), (13, 10), (13, 11), (13, 12), (13, 13), (13, 14), (13, 15), (13, 16), (13, 17), (13, 18)]
        for idx, file in enumerate(test_f_list):
            sweep = pickle.load(open(file, "rb"))
            for cell_pos, k in enumerate(sweep.keys()):
                if k in select_cells:
                    self.test_cleaned_velo.append(sweep[k])

        dps = np.concatenate(self.test_cleaned_velo, 0)
        
        d_num = 200000
        idx = np.random.choice(dps.shape[0], d_num, replace=False)
        dps = dps[idx]
        center = self._get_mean(dps)
        dps -= center

        # random.seed(10)
        # self.sample_idx = randint(0, len(test_f_list)-1)
        # self.sample_sweep = []
        # self.sample_points = self.partition_hdmap([test_f_list[self.sample_idx]], self.sample_sweep, permutation=False)

    def partition(self, train_cleaned_velo):
        valid_split = int(len(train_cleaned_velo)*0.9)
        train_velo = train_cleaned_velo
        if self.fb_split == True:
            print ('partitioning training set...')
            train_cell_f, train_cell_b = self.fb_partition_batch(train_velo[:valid_split])
            print ('partitioning validation set...')
            valid_cell_f, valid_cell_b = self.fb_partition_batch(train_velo[valid_split:])
            
            print ("total foreground training set: {}, validation set: {}, testing set: {}".format(train_cell_f[0].shape[0], 
                                                                                                   valid_cell_f[0].shape[0],
                                                                                                   self.test_cell_f[0].shape[0]))  
            print ("total background training set: {}, validation set: {}, testing set: {}".format(train_cell_b[0].shape[0], 
                                                                                                   valid_cell_b[0].shape[0],
                                                                                                   self.test_cell_b[0].shape[0]))
        else:
            print ('partitioning training set...')
            train_cell = self.partition_batch(train_velo[:valid_split])
            print ('partitioning validation set...')
            valid_cell = self.partition_batch(train_velo[valid_split:])
            
            print ("total training set: {}, validation set: {}, testing set: {}".format(train_cell[0].shape[0], 
                                                                                        valid_cell[0].shape[0],
                                                                                        self.test_cell[0].shape[0]))
        print ("batch size: {}".format(self.batch_size))
        return train_cell, valid_cell

    def fb_partition_batch(self, cleaned_velo):

        foreground, background = [], []
        for points in cleaned_velo:
            f_points, b_points = rough_LBS(points)
            foreground.append(f_points)
            background.append(b_points)

        data_cell_f = self.partition_batch(foreground)

        data_cell_b = self.partition_batch(background)

        return data_cell_f, data_cell_b

    def partition_batch(self, cleaned_velo, permutation=False):
        cell_points = [[] for _ in range(self.level)]
        if self.range_view == True:
            for idx, points in enumerate(tqdm(cleaned_velo)):
                cell_points_buf = self.partition_single_range(points, idx)
                for l in range(self.level):
                    cell_points[l].append(cell_points_buf[l])
        else:
            for idx, points in enumerate(tqdm(cleaned_velo)):
                cell_points_buf = self.partition_single(points, idx)
                for l in range(self.level):
                    cell_points[l].append(cell_points_buf[l])

        data_cell = []
        for l in range(self.level):
            cell_points[l] = pd.concat(cell_points[l])
            self.cell_set[l] = cell_points[l]

            data_buf = cell_points[l]
            if permutation == True:
                data_buf = data_buf.iloc[np.random.permutation(np.arange(len(data_buf)))]
            data_cell.append(data_buf)

        return data_cell

    def partition_single(self, points, idx):
        # xyz_max = np.max(points, 0)
        # xyz_min = np.min(points, 0)
        xyz_max = np.array([self.max_x_dist, self.max_y_dist, self.max_z_dist])
        xyz_min = np.array([-self.max_x_dist, -self.max_y_dist, -self.max_z_dist])
        
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
                    cell_pos = i*W+j

                    if num_points < self.cell_min_points[k]:
                        cell_points.iloc[cell_pos]['compressed'] = False
                        # continue ## testing, need to removed later
                    else:
                        cell_points.iloc[cell_pos]['compressed'] = True

                    if num_points == 0:
                        cell, ratio, center = np.zeros([self.cell_max_points[k], 3]), 1.0, np.zeros(3)
                    elif num_points > self.cell_max_points[k]: # if greater than max points, random sample
                        sample_index = np.random.choice(num_points, self.cell_max_points[k], replace=False)
                        cell, ratio, center = rescale(points[cell_idx][sample_index], self.cell_max_points[k])
                    elif num_points <= self.cell_max_points[k]:
                        new_point = np.zeros([self.cell_max_points[k], 3])
                        new_point[:num_points] = points[cell_idx]
                        cell, ratio, center = rescale(new_point, num_points)

                    cell_points.iloc[cell_pos]['points'] = cell.astype(np.float32)
                    # if k > 0:
                    #     I, J = i//self.factor, j//self.factor
                    #     parent_pos = int(I*self.W[k-1]+J)
                    #     cell_points.iloc[cell_pos]['stacked'] = multi_cell[-1].iloc[parent_pos]['points']
                    #     cell_points.iloc[cell_pos]['stacked_num'] = multi_cell[-1].iloc[parent_pos]['num_points']
                    cell_points.iloc[cell_pos]['index'] = idx
                    cell_points.iloc[cell_pos]['ratio'] = ratio
                    cell_points.iloc[cell_pos]['center'] = center
                    cell_points.iloc[cell_pos]['row'] = i
                    cell_points.iloc[cell_pos]['col'] = j
                    cell_points.iloc[cell_pos]['level'] = k
                    cell_points.iloc[cell_pos]['num_points'] = min(num_points, self.cell_max_points[k])
                    cell_points.iloc[cell_pos]['stacked'] = cell.astype(np.float32)
                    cell_points.iloc[cell_pos]['stacked_num'] = min(num_points, self.cell_max_points[k])
            multi_cell.append(cell_points)

        for k in range(self.level):
            multi_cell[k] = multi_cell[k].dropna(thresh=5)
        return multi_cell

    def compute_range_view_image_pixel_id(self, point_cloud, level_idx=0):
        '''
        input: point_cloud:  N * 3 point cloud matrix
        
        output: row_id:    N * 1 row id
                column_id: N * 1 column id
        '''
        x_coords = point_cloud[:, 0]
        y_coords = point_cloud[:, 1]
        z_coords = point_cloud[:, 2]
        range_xy = np.sqrt(np.square(x_coords)+np.square(y_coords))
        elevate_angle = np.arctan2(z_coords, range_xy) * 180.0 /np.pi
        azimuth_angle = np.arctan2(y_coords, x_coords) * 180.0 /np.pi

        # compute row id
        lower_row_id = np.floor((elevate_angle - self.lower_min_elevate_angle)*2)
        upper_row_id = np.floor((elevate_angle - self.upper_min_elevate_angle)*3) + 32
        row_id = np.where(elevate_angle < self.upper_min_elevate_angle, lower_row_id, upper_row_id)
        row_id[row_id < 0] = 0
        row_id[row_id >= self.num_lasers] = self.num_lasers - 1
        row_id = row_id / self.group_num_lasers[level_idx]

        # compute row id
        inverse_range_of_azimuth_angle = 1.0 / self.range_of_azimuth_angle
        column_id = np.floor((azimuth_angle - self.min_azimuth_angle) * self.image_width[level_idx] * inverse_range_of_azimuth_angle)
        column_id[column_id==self.image_width[level_idx]] = 0

        return row_id.astype(int), column_id.astype(int)

    def partition_single_range(self, points, idx):
        '''
        input: point_cloud:  N * 3 point cloud matrix
        
        output: range_view_cell_points:    IMAGE_HEIGHT, IMAGE_WIDTH, MAX_POINTS_IN_A_CELL, 4 (x,y,z,intensity)
                range_view_cell_point_num: IMAGE_HEIGHT * IMAGE_WIDTH
        '''

        # # Create a dictionary of cell information
        # cell_info = dict()
        # for i in range(len(points)):
        #     key = (r[i], c[i])
        #     if key in cell_info:
        #         cell_info[key].append(points[i])
        #     else:
        #         cell_info[key] = [points[i]]

        multi_cell = []
        for k in range(self.level):
            r, c = self.compute_range_view_image_pixel_id(points, k)
            index = np.arange(self.image_height[k]*self.image_width[k])
            cell_points = pd.DataFrame(index=index, columns=self.columns)
            for i in range(self.image_height[k]):
                for j in range(self.image_width[k]):
                    cell_idx = np.intersect1d(np.where(r == i)[0], np.where(c == j)[0])
                    num_points = cell_idx.shape[0]
                    cell_pos = i*self.image_width[k] + j

                    if num_points < self.cell_min_points[0]:
                        cell_points.iloc[cell_pos]['compressed'] = False
                    else:
                        cell_points.iloc[cell_pos]['compressed'] = True

                    if num_points == 0:
                        cell, ratio, center = np.zeros([self.cell_max_points[k], 3]), 1.0, np.zeros(3)
                    elif num_points > self.cell_max_points[k]: # if greater than max points, random sample
                        sample_index = np.random.choice(num_points, self.cell_max_points[k], replace=False)
                        cell, ratio, center = rescale(points[cell_idx][sample_index], self.cell_max_points[k])
                    elif num_points <= self.cell_max_points[k]:
                        new_point = np.zeros([self.cell_max_points[k], 3])
                        new_point[:num_points] = points[cell_idx]
                        cell, ratio, center = rescale(new_point, num_points)

                    # if k > 0:
                    #     I, J = i//self.factor, j//self.factor
                    #     parent_pos = int(I*self.W[k-1]+J)
                    #     cell_points.iloc[cell_pos]['stacked'] = multi_cell[-1].iloc[parent_pos]['points']
                    #     cell_points.iloc[cell_pos]['stacked_num'] = multi_cell[-1].iloc[parent_pos]['num_points']
                    cell_points.iloc[cell_pos]['points'] = cell.astype(np.float32)
                    cell_points.iloc[cell_pos]['row'] = i
                    cell_points.iloc[cell_pos]['col'] = j
                    cell_points.iloc[cell_pos]['index'] = idx
                    cell_points.iloc[cell_pos]['ratio'] = ratio
                    cell_points.iloc[cell_pos]['center'] = center
                    cell_points.iloc[cell_pos]['level'] = k
                    cell_points.iloc[cell_pos]['num_points'] = min(num_points, self.cell_max_points[k])
                    cell_points.iloc[cell_pos]['stacked'] = cell.astype(np.float32)
                    cell_points.iloc[cell_pos]['stacked_num'] = min(num_points, self.cell_max_points[k])
                    
            multi_cell.append(cell_points)

        for k in range(self.level):
            multi_cell[k] = multi_cell[k].dropna(thresh=5)
        return multi_cell

    def test_compression_rate(self, level_idx=0):

        print ('calculating average testing compression rate ...')
        cell_rate_record, sweep_rate_record = [], []
        for i, sweep in enumerate(self.test_sweep):
            if self.range_view == True:
                sweep_cell = self.partition_single_range(sweep, i)
            else:
                sweep_cell = self.partition_single(sweep, i)
            cell_rate, sweep_rate = self.calculate_compression_rate(sweep_cell[level_idx]) # only calculate compression on the finest scale
            cell_rate_record.append(cell_rate)
            sweep_rate_record.append(sweep_rate)
        print ('average cell compression rate: %.6f' % (np.mean(cell_rate_record)))
        print ('average sweep compression rate: %.6f' % (np.mean(sweep_rate_record)))
        return np.mean(cell_rate_record), np.mean(sweep_rate_record)

    def calculate_compression_rate(self, sample_points):
        compressed_cell = sample_points[sample_points['compressed']==True]
        compressed_num = compressed_cell['num_points'].sum() * 3
        compressed_cell_num = compressed_cell.shape[0]
        latent_num = self.latent_dim * compressed_cell_num

        original_cell = sample_points[sample_points['compressed']==False]
        original_num = original_cell['num_points'].sum() * 3
        total_compressed = latent_num + original_num
        total_original = compressed_num + original_num
        return latent_num/compressed_num, total_compressed/total_original

    def reconstruct_scene(self, cell_points, cell_info, stacked=False):
        scene = []
        for i in range(len(cell_info)):
            cell = cell_points[i]
            if stacked == True:
                num_points = cell_info.iloc[i]['stacked_num'][-1]
            else:
                num_points = cell_info.iloc[i]['num_points']
            scaled_cell = cell[:num_points] / cell_info.iloc[i]['ratio']
            scaled_cell += cell_info.iloc[i]['center']
            scene.append(scaled_cell)
        return np.concatenate(scene, axis=0)

    def all_data(self):
        return self.cell_points




# class LoadData():
#     def __init__(self, args, train_drives, test_drives):
        
#         self.batch_size = args.batch_size
#         self.batch_idx = 0
#         self.basedir = args.data_in_dir
#         self.L = list(map(int, args.PL.split(',')))
#         self.W = list(map(int, args.PL.split(',')))
#         self.factor = self.L[1] / self.L[0] if len(self.L) > 1 else 1
#         self.level = len(self.L)
#         self.fb_split = args.fb_split
#         self.cell_max_points = list(map(int, args.cell_max_points.split(',')))
#         self.cell_min_points = list(map(int, args.cell_min_points.split(',')))
#         self.max_x_dist = 16.0
#         self.max_y_dist = 16.0
#         self.max_z_dist = 3.0

#         self.columns = ['points', 'children', 'index', 'ratio', 'center', 'level', 'num_points']
#         if self.level > 1:
#             self.cell_set = []
#             for i in range(self.level):
#                 self.cell_set.append(pd.DataFrame(columns=self.columns))
#         else:
#             self.cell_set = [pd.DataFrame(columns=self.columns)]

#         # processing training drives
#         self.data_id = []
#         self.train_dataset_gray, self.train_dataset_rgb, self.train_dataset_velo = [], [], []
#         self.test_dataset_gray, self.test_dataset_rgb, self.test_dataset_velo = [], [], []
#         num_to_select = 10
#         for drive in train_drives:
#             dataset = load_dataset(args.date, drive, self.basedir)
#             self.data_id += [args.date + '_' + drive]
#             data_len = len(list(dataset.velo))
#             Idx = random.sample(range(data_len), num_to_select)
#             # for visualization
#             # tracklet_rects, tracklet_types = load_tracklets_for_frames(len(list(dataset.velo)), '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(basedir, date, date, drive))
#             # frame = 10 # for visualization
#             # display_frame_statistics(dataset, tracklet_rects, tracklet_types, frame)
#             self.train_dataset_gray += [list(dataset.gray)[i] for i in Idx]
#             self.train_dataset_rgb += [list(dataset.rgb)[i] for i in Idx]
#             self.train_dataset_velo += [list(dataset.velo)[i] for i in Idx]
        
#         self.train_cleaned_velo = []
#         for points in self.train_dataset_velo:
#             idx_z = np.squeeze(np.argwhere(abs(points[:,2]) > self.max_z_dist)) # remove all points below -3.0 in z-axis
#             idx_x = np.squeeze(np.argwhere(abs(points[:,0]) > self.max_x_dist)) # remove all points above 32m
#             idx_y = np.squeeze(np.argwhere(abs(points[:,1]) > self.max_y_dist)) # remove all points above 32m
#             idx = np.union1d(np.union1d(idx_x, idx_y), idx_z)
#             filter_data = np.delete(points, idx, axis=0)[:,:3]
#             # testing
#             # index = random.sample(range(0, len(filter_data)-1), 20000)
#             # self.cleaned_velo.append(filter_data[index])
#             self.train_cleaned_velo.append(filter_data) # ignore the 4th dimension

#         for drive in test_drives:
#             dataset = load_dataset(args.date, drive, self.basedir)
#             self.data_id += [args.date + '_' + drive]
#             data_len = len(list(dataset.velo))
#             Idx = random.sample(range(data_len), num_to_select)
#             self.test_dataset_gray += [list(dataset.gray)[i] for i in Idx]
#             self.test_dataset_rgb += [list(dataset.rgb)[i] for i in Idx]
#             self.test_dataset_velo += [list(dataset.velo)[i] for i in Idx]

#         self.test_cleaned_velo = []
#         for points in self.test_dataset_velo:
#             idx_z = np.squeeze(np.argwhere(abs(points[:,2]) > self.max_z_dist)) # remove all points below -3.0 in z-axis
#             idx_x = np.squeeze(np.argwhere(abs(points[:,0]) > self.max_x_dist)) # remove all points above 32m
#             idx_y = np.squeeze(np.argwhere(abs(points[:,1]) > self.max_y_dist)) # remove all points above 32m
#             idx = np.union1d(np.union1d(idx_x, idx_y), idx_z)
#             filter_data = np.delete(points, idx, axis=0)[:,:3]
#             self.test_cleaned_velo.append(filter_data) # ignore the 4th dimension

#         self.test_sweep = self.test_cleaned_velo
#         valid_split = int(len(self.train_cleaned_velo)*0.9)
#         if self.fb_split == True:
#             print ('partitioning training set...')
#             self.train_cell_f, self.train_cell_b = self.fb_partition_batch(self.train_cleaned_velo[:valid_split])
#             print ('partitioning validation set...')
#             self.valid_cell_f, self.valid_cell_b = self.fb_partition_batch(self.train_cleaned_velo[valid_split:])
#             print ('partitioning testing set...')
#             self.test_cell_f, self.test_cell_b = self.fb_partition_batch(self.test_cleaned_velo)
            
#             print ("total foreground training set: {}, validation set: {}, testing set: {}".format(self.train_cell_f[0].shape[0], 
#                                                                                                    self.valid_cell_f[0].shape[0],
#                                                                                                    self.test_cell_f[0].shape[0]))  
#             print ("total background training set: {}, validation set: {}, testing set: {}".format(self.train_cell_b[0].shape[0], 
#                                                                                                    self.valid_cell_b[0].shape[0],
#                                                                                                    self.test_cell_b[0].shape[0]))

#         else:
#             print ('partitioning training set...')
#             self.train_cell = self.partition_batch(self.train_cleaned_velo[:valid_split])
#             print ('partitioning validation set...')
#             self.valid_cell = self.partition_batch(self.train_cleaned_velo[valid_split:])
#             print ('partitioning testing set...')
#             self.test_cell = self.partition_batch(self.test_cleaned_velo)

#             print ("total training set: {}, validation set: {}, testing set: {}".format(self.train_cell[0].shape[0], 
#                                                                                         self.valid_cell[0].shape[0],
#                                                                                         self.test_cell[0].shape[0]))
#         print ("batch size: {}".format(self.batch_size))

#         # sample
#         self.sample_idx = randint(0, len(self.test_sweep)-1)
#         self.sample_idx_train = randint(0, len(self.train_cleaned_velo)-1)

#         if self.fb_split == True:
#             f_points, b_points = rough_LBS(self.test_sweep[self.sample_idx])
#             self.test_f_sweep, self.test_b_sweep = f_points, b_points
#             self.sample_points_f = self.partition_single(f_points, self.sample_idx)[0]
#             self.sample_points_b = self.partition_single(b_points, self.sample_idx)[0]
#             # training set, see if overfitting
#             f_points_t, b_points_t = rough_LBS(self.train_cleaned_velo[self.sample_idx_train])
#             self.train_f_sweep, self.train_b_sweep = f_points_t, b_points_t
#             self.train_points_f = self.partition_single(f_points_t, self.sample_idx_train)[0]
#             self.train_points_b = self.partition_single(b_points_t, self.sample_idx_train)[0]
            
#         else:
#             self.sample_points = self.partition_single(self.test_sweep[self.sample_idx], self.sample_idx)[0]
#             # training set, see if overfitting
#             self.train_points = self.partition_single(self.train_cleaned_velo[self.sample_idx_train], self.sample_idx_train)[0]


#     def fb_partition_batch(self, cleaned_velo):

#         foreground, background = [], []
#         for points in cleaned_velo:
#             f_points, b_points = rough_LBS(points)
#             foreground.append(f_points)
#             background.append(b_points)

#         data_cell_f = self.partition_batch(foreground)

#         data_cell_b = self.partition_batch(background)

#         return data_cell_f, data_cell_b

#     def partition_batch(self, cleaned_velo):
#         cell_points = [[] for _ in range(self.level)]
#         for idx, points in enumerate(tqdm(cleaned_velo)):
#             for l in range(self.level):
#                 cell_points_buf = self.partition_single(points, idx)
#                 cell_points[l].append(cell_points_buf[l])

#         data_cell = []
#         for l in range(self.level):
#             cell_points[l] = pd.concat(cell_points[l])
#             self.cell_set[l] = cell_points[l]

#             data_buf = cell_points[l]
#             data_buf = data_buf.iloc[np.random.permutation(np.arange(len(data_buf)))]
#             data_cell.append(data_buf)

#         return data_cell
        
#     def partition_single(self, points, idx):
#         xyz_max = np.max(points, 0)
#         xyz_min = np.min(points, 0)
        
#         multi_cell = []
#         # partition into a LxW grid
#         for k in range(self.level-1, -1, -1):
#             L, W = self.L[k], self.W[k]
#             cell_width, cell_length = (xyz_max[0]-xyz_min[0])/W, (xyz_max[1]-xyz_min[1])/L
#             index = np.arange(L*W)
#             cell_points = pd.DataFrame(index=index, columns=self.columns)
#             for i in range(L):
#                 for j in range(W):
#                     left, right = xyz_min[0]+i*cell_width, xyz_min[0]+(i+1)*cell_width 
#                     up, down = xyz_min[1]+j*cell_length, xyz_min[1]+(j+1)*cell_length
#                     cell_idx = np.where(np.logical_and(np.logical_and(points[:,0]>=left, points[:,0]<right), np.logical_and(points[:,1]>=up, points[:,1]<down)))[0]
#                     num_points = len(cell_idx)

#                     if num_points < self.cell_min_points[k]:
#                         continue

#                     if num_points > self.cell_max_points[k]: # if greater than max points, random sample
#                         sample_index = np.random.choice(num_points, self.cell_max_points[k], replace=False)
#                         cell, ratio, center = rescale(points[cell_idx][sample_index], self.cell_max_points[k])
#                     elif num_points <= self.cell_max_points[k]:
#                         new_point = np.zeros([self.cell_max_points[k], 3])
#                         new_point[:num_points] = points[cell_idx]
#                         cell, ratio, center = rescale(new_point, num_points)

#                     cell_idx = i*W+j
#                     cell_points.iloc[cell_idx]['points'] = cell
#                     if k < self.level-1:
#                         children = []
#                         I, J = i*self.factor, j*self.factor
#                         dij = []
#                         for r in range(int(self.factor)):
#                             for c in range(int(self.factor)):
#                                 children_idx = int((I+r)*self.W[k+1]+(J+c))
#                                 ch = multi_cell[-1].iloc[children_idx]['points']
#                                 if ch != np.nan:
#                                     children.append(ch)
#                         cell_points.iloc[cell_idx]['children'] = np.array(children)
#                     cell_points.iloc[cell_idx]['index'] = idx
#                     cell_points.iloc[cell_idx]['ratio'] = ratio
#                     cell_points.iloc[cell_idx]['center'] = center
#                     cell_points.iloc[cell_idx]['level'] = k
#                     cell_points.iloc[cell_idx]['num_points'] = min(num_points, self.cell_max_points[k])
#             multi_cell.append(cell_points)

#         for k in range(self.level):
#             multi_cell[k] = multi_cell[k].dropna(thresh=2)
#         return multi_cell

#     def reconstruct_scene(self, cell_points, cell_info):
#         scene = []
#         for i in range(len(cell_info)):
#             cell = cell_points[i]
#             num_points = cell_info.iloc[i]['num_points']
#             scaled_cell = cell[:num_points] / cell_info.iloc[i]['ratio']
#             scaled_cell += cell_info.iloc[i]['center']
#             scene.append(scaled_cell)
#         return np.concatenate(scene, axis=0)

#     def all_data(self):
#         return self.cell_points



    
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

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# def visualize_3d_points(dataset, points=1.0, elev=40, azim=75, dir='.', filename='sample'):
#     """
#     Displays statistics for a single frame. 3D plot of the lidar point cloud data and point cloud
#     projections to various planes.
    
#     Parameters
#     ----------
#     dataset         : `raw` dataset.
#     points          : Fraction of lidar points to use. Defaults to `0.2`, e.g. 20%.
#     """

#     dataset_velo = dataset
    
#     points_step = int(1. / points)
#     point_size = 0.05 * (1. / points)
#     velo_range = range(0, dataset_velo.shape[0], points_step)
#     velo_frame = dataset_velo[velo_range, :]      
#     def draw_point_cloud(ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
#         """
#         Convenient method for drawing various point cloud projections as a part of frame statistics.
#         """
#         ax.scatter(*np.transpose(velo_frame[:, axes]), s=point_size, cmap='gray')
#         ax.set_title(title)
#         ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
#         ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
#         if len(axes) > 2:
#             ax.set_xlim3d(*axes_limits[axes[0]])
#             ax.set_ylim3d(*axes_limits[axes[1]])
#             ax.set_zlim3d(*axes_limits[axes[2]])
#             ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
#         else:
#             # ax.set_xlim(-30, 30)
#             # ax.set_ylim(-30, 30)
#             ax.set_xlim(*axes_limits[axes[0]])
#             ax.set_ylim(*axes_limits[axes[1]])
#         # User specified limits
#         if xlim3d!=None:
#             ax.set_xlim3d(xlim3d)
#         if ylim3d!=None:
#             ax.set_ylim3d(ylim3d)
#         if zlim3d!=None:
#             ax.set_zlim3d(zlim3d)
                        
#     # Draw point cloud data as 3D plot
#     f2 = plt.figure(figsize=(15, 8))
#     ax2 = f2.add_subplot(111, projection='3d')
#     ax2.axes.set_aspect('equal')
#     ax2.view_init(elev=elev, azim=azim)
#     draw_point_cloud(ax2, 'Velodyne scan')
#     # draw_point_cloud(ax2, 'Velodyne scan', xlim3d=(-40,40), ylim3d=(-100,100), zlim3d=(-2,10))
#     # draw_point_cloud(ax2, 'Velodyne scan', xlim3d=(-20,20))
#     # set_axes_equal(ax2)
#     plt.savefig(dir+'/3d_point_cloud_'+filename)
    
#     # Draw point cloud data as plane projections
#     f, ax3 = plt.subplots(3, 1, figsize=(15, 25))
#     draw_point_cloud(
#         ax3[0], 
#         'Velodyne scan, XZ projection (Y = 0), the car is moving in direction left to right', 
#         axes=[0, 2] # X and Z axes
#     )
#     draw_point_cloud(
#         ax3[1], 
#         'Velodyne scan, XY projection (Z = 0), the car is moving in direction left to right', 
#         axes=[0, 1] # X and Y axes
#     )
#     draw_point_cloud(
#         ax3[2], 
#         'Velodyne scan, YZ projection (X = 0), the car is moving towards the graph plane', 
#         axes=[1, 2] # Y and Z axes
#     )
#     plt.savefig(dir+'/2d_point_cloud_'+filename)

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
        # ax.grid(False)
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
    ax2 = Axes3D(f2)
    # ax2.view_init(elev=20)
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
    # # has memory overflow issue
    # dist = scipy.spatial.distance.cdist(orig, recon) # order must be recon, orig
    # dist_vec = np.min(dist, 1)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(recon)
    dist_vec, _ = nbrs.kneighbors(orig)
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


