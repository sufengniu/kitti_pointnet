import numpy as np
import tensorflow as tf
import os
import os.path as osp
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from util import *

cell_per_meter = 5
cellmap_radius = 30
cellmap_size = 2*cell_per_meter*cellmap_radius


def cell_cut(velo_frame):
    front = np.argwhere(velo_frame[:,2] >= -1)
    front_point = np.squeeze(velo_frame[front,:], 1)
    back = np.argwhere(velo_frame[:,2] < -1)
    back_point = np.squeeze(velo_frame[back,:], 1) # back point no used???
    
    # front point
    x_min, x_max = np.min(front_point[:,0]), np.max(front_point[:,0])
    y_min, y_max = np.min(front_point[:,1]), np.max(front_point[:,1])
    
    cell_point = []
    for i in range((x_max-x_min)//cell_per_meter):
        for j in range((y_max-y_min)//cell_per_meter):
            row_selected = front_point[np.where(front_point[:,0]>i*cell_per_meter)[0]]
            row_selected = front_point[np.where(row_selected[:,0]<(i+1)*cell_per_meter)[0]]
            col_selected = front_point[np.where(front_point[:,1]>j*cell_per_meter)[0]]
            col_selected = front_point[np.where(row_selected[:,1]<(j+1)*cell_per_meter)[0]]
            cell_point.append(col_selected)
    return cell_point


def generate_pc_data(batch_size):
    date = '2011_09_26'
    drive = '0001'
    basedir = '/scratch2/sniu/kitti'

    dataset = load_dataset(date, drive, basedir)
    # for visualization
    tracklet_rects, tracklet_types = load_tracklets_for_frames(len(list(dataset.velo)), '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(basedir, date, date, drive))
    frame = 10 # for visualization
    display_frame_statistics(dataset, tracklet_rects, tracklet_types, frame)

    dataset_velo = list(dataset.velo)
    cut_point = cell_cut(dataset_velo[0])

    sample_point = dataset_velo[0]
    
    return sample_point, cut_point


