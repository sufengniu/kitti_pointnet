import numpy as np
import tensorflow as tf
import os
import os.path as osp
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from multiprocessing import Pool

from util import *


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

def get_cellmap_from_points(velo_frame, cell_per_meter=5, cellmap_radius=30):
    
    cellmap_size = 2*cell_per_meter*cellmap_radius

    foreground_index = np.argwhere(velo_frame[:,2] >= -1)
    foreground_points = np.squeeze(velo_frame[foreground_index,:], 1)
    backgroud_index = np.argwhere(velo_frame[:,2] < -1)
    backgroud_points = np.squeeze(velo_frame[backgroud_index,:], 1)
    
    # Extract foreground points within cellmap
    index_within_xm = (abs(foreground_points[:, 0]) < cellmap_radius) &  (abs(foreground_points[:, 1]) < cellmap_radius);
    foreground_points_within_cellmap = foreground_points[index_within_xm, :]
    cell_id_2d = cellmap_radius*cell_per_meter + np.ceil(foreground_points_within_cellmap[:, 0:2]*cell_per_meter)

    cell_id = sub2ind([cellmap_size, cellmap_size], cell_id_2d[:,0], cell_id_2d[:,1])
    cell_id_unique = np.unique(cell_id) 

    cellmap = {}
    for i_cell in cell_id_unique:
        cell_id_index = (cell_id == i_cell)
        cellmap[i_cell] = foreground_points_within_cellmap[cell_id_index, :]

    return cellmap

def get_points_from_cellmap(cellmap, cell_per_meter=5, cellmap_radius=30, num_points_lower_threshold = 50):

    points = None
    for i_cell in cellmap.keys():
        if points == None:
            points = cellmap[i_cell]
        else:
            points = np.concatenate([points, cellmap[i_cell]], axis=0)
    return points


def get_batch_data_from_cellmap(cellmap, num_points_lower_threshold=50, num_points_upper_threshold=100):

    batch_data = []
    meta_data = []

    for i_cell in cellmap.keys():
        points = cellmap[i_cell]
        num_points = points.shape[0]
        if num_points < num_points_lower_threshold:
            continue

        meta_data.append(num_points)
        if num_points == num_points_upper_threshold:
            batch_data.append(points)

        if num_points < num_points_upper_threshold:
            tmp = np.concatenate([points, np.zeros((num_points_upper_threshold-num_points, 4))],axis=0)
            batch_data.append(tmp)

        if num_points > num_points_upper_threshold:
            sample_index = np.random.choice(num_points, num_points_upper_threshold, replace=False)
            batch_data.append(points[sample_index, :])

    # batch * num_points_upper_threshold * 4
    batch_data = np.stack(batch_data)
    # batch
    meta_data = np.stack(meta_data)

    return batch_data, meta_data

def shuffle_data(point_clouds, meta, seed=12345):
    num_examples = point_clouds.shape[0]
    if seed is not None:
        np.random.seed(seed)
    perm = np.arange(num_examples)
    np.random.shuffle(perm)
    return point_clouds[perm], meta[perm]

def rescale(point_clouds, meta, num_points_upper_threshold=100):


    point_mean = []
    point_scaled = []
    for i in range(point_clouds.shape[0]):
        p_scaled = point_clouds[i, np.arange(min(meta[i], num_points_upper_threshold)), :]
        x_max, y_max, z_max = np.max(p_scaled, 0)
        x_min, y_min, z_min = np.min(p_scaled, 0)
        x_var, y_var, z_var = x_max-x_min, y_max-y_min, z_max-z_min
        x_mean, y_mean, z_mean = np.mean(p_scaled, 0)
        point_mean.append(np.array([x_mean, y_mean, z_mean]))
        point_clouds[i, np.arange(min(meta[i], num_points_upper_threshold)), :] = p_scaled - np.array([x_mean, y_mean, z_mean])
        point_scaled.append(point_clouds[i,:,:])
        # point_clouds[i, np.arange(meta[i]), :] = point_clouds[i, np.arange(meta[i]), :] / np.array([x_var, y_var, z_var])
    point_mean = np.stack(point_mean)
    point_scaled = np.stack(point_scaled)
    return point_scaled, point_mean


def load_pc_data(batch_size, n_threads=16, shuffle=True):
    date = '2011_09_26'
    drive = '0001'
    basedir = '/scratch2/sniu/kitti'
    num_points_upper_threshold = 100

    dataset = load_dataset(date, drive, basedir)
    # for visualization
    # tracklet_rects, tracklet_types = load_tracklets_for_frames(len(list(dataset.velo)), '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(basedir, date, date, drive))
    # frame = 10 # for visualization
    # display_frame_statistics(dataset, tracklet_rects, tracklet_types, frame)

    dataset_velo = list(dataset.velo)
    all_points = np.empty([0, num_points_upper_threshold, dataset_velo[0].shape[1]], dtype=np.float32)
    all_meta = np.empty([0], dtype=np.int32)

    print ('------ loading pointnet data ------')
    pool = Pool(n_threads)
    for i, cellmap in enumerate(pool.imap(get_cellmap_from_points, dataset_velo)):
        batch_points, batch_meta = get_batch_data_from_cellmap(cellmap, num_points_upper_threshold=num_points_upper_threshold)

        all_points = np.concatenate([all_points, batch_points], axis=0)
        all_meta = np.concatenate([all_meta, batch_meta], axis=0)
        
    pool.close()
    pool.join()

    if shuffle == True:
        all_points, all_meta = shuffle_data(all_points, all_meta)

    all_points = all_points[:,:,:3]
    all_points, all_mean = rescale(all_points, all_meta, num_points_upper_threshold=num_points_upper_threshold)

    return all_points, all_meta, all_mean


def generate_batch(batch_size, all_points, all_meta):

