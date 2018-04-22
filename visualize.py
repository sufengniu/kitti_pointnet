import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_util
import data
import os
import os.path as osp

from sklearn import linear_model, datasets
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn import mixture

from external.structural_losses.tf_nndistance import nn_distance
from external.structural_losses.tf_approxmatch import approx_match, match_cost

FLAGS = tf.flags.FLAGS

# Optimizer parameters.
tf.flags.DEFINE_float("seed", 3122018, "Random seeds for results reproduction")

# Training options.
tf.flags.DEFINE_string("top_out_dir", "/scratch3/sniu/point_cloud/checkpoints",
                       "Checkpointing directory.")
tf.flags.DEFINE_string("top_in_dir", "/scratch2/sniu/point_cloud/shape_net_core_uniform_samples_2048/",
                       "data input dir")
tf.flags.DEFINE_string("encoder", "point",
                       "encoder type [edge|point]")
tf.flags.DEFINE_string("decoder", "fc",
                       "decoder type [fc|grid]")
tf.flags.DEFINE_integer("split_mode", 3,
                       "0: foreground, 1: background, 2: all points, 3: clustering on foreground")
tf.flags.DEFINE_string("cluster_mode", 'kmeans',
                       "kmeans, spectral, dirichlet")
tf.flags.DEFINE_integer("num_points", 200,
                       "number of points in point cloud")
tf.flags.DEFINE_integer("batch_size", 512,
                       "batch size")

batch_size = FLAGS.batch_size
origin_num_points = 1024 if FLAGS.split_mode==3 else FLAGS.num_points
points = 0.8
frame = 10
num_cluster = 64

# visualization limits
axes_limits = [
    [-20, 80], # X axis range
    [-20, 20], # Y axis range
    [-3, 10]   # Z axis range
]
axes_str = ['X', 'Y', 'Z']

dataset_velo, dataset_gray, dataset_rgb, tracklet_rects, tracklet_types = data.load_orig_pc_data(batch_size, train=False)

def LBS_idx(point):
    ransac = linear_model.RANSACRegressor()
    ransac.fit(point[:,0:2], point[:,2])
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    return inlier_mask, outlier_mask

def cluster_func(raw_points, n_threads=16, cluster_mode="spectral"):
    # affinity = 'rbf'
    affinity = 'nearest_neighbors'
    num_neighbors = 20

    # seeds = [1,22123,3010,4165,54329,62,7656,80109212] # for better generalization
     # for better generalization
    cluster_idx = range(num_cluster)
    cellmap = {}
    if cluster_mode == "dirichlet":
        spectral = mixture.BayesianGaussianMixture(n_components=num_cluster, covariance_type='full', random_state=1).fit(raw_points)
        cluster_idx = np.unique(spectral.predict(raw_points))
    else:
        if cluster_mode == 'kmeans':
            spectral = KMeans(n_clusters=num_cluster, random_state=1, n_jobs=n_threads).fit(raw_points)
        elif cluster_mode == "spectral":
            spectral = SpectralClustering(num_cluster, random_state=1, affinity=affinity, n_neighbors=num_neighbors, n_jobs=n_threads).fit(raw_points)
    for i, index in enumerate(cluster_idx):
        if hasattr(spectral, 'labels_'):
            idx = np.where(spectral.labels_.astype(np.int)==index)[0]
        else:
            idx = np.where(spectral.predict(raw_points)==index)[0]
        cellmap[i] = idx
    return cellmap

def draw_point_cloud(ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
    """
    Convenient method for drawing various point cloud projections as a part of frame statistics.
    """
    ax.scatter(*np.transpose(velo_frame[:, axes]), s=point_size, c=velo_frame[:, 3])
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


# plot
points_step = int(1. / points)
point_size = 0.01 * (1. / points)
velo_range = range(0, dataset_velo[frame].shape[0], points_step)
velo_frame = dataset_velo[frame][velo_range, :]

background_idx, foreground_idx = LBS_idx(velo_frame)

velo_frame[background_idx, 3] = 0.9
cellmap = cluster_func(velo_frame[foreground_idx, 0:3], cluster_mode='spectral')
for i in range(num_cluster):
    velo_frame[cellmap[i], :3] = i/100.0

# Draw point cloud data as 3D plot
f2 = plt.figure(figsize=(15, 8))
ax2 = f2.add_subplot(111, projection='3d')
draw_point_cloud(ax2, 'Velodyne scan', xlim3d=(-10,30))
plt.savefig('3d_point_cloud_0005')