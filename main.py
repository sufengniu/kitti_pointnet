import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import sys
import numpy as np
import tensorflow as tf
import os
import os.path as osp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR = '/home/sniu/lab/ai_lab/pc_pool/kitti_pointnet'
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from model import AutoEncoder
import tf_util
import util

from external.structural_losses.tf_nndistance import nn_distance
from external.structural_losses.tf_approxmatch import approx_match, match_cost

FLAGS = tf.flags.FLAGS

# Training options.
tf.flags.DEFINE_string("save_path", "/scratch4/sniu/pointnet_full_emd/",
                       "Checkpointing directory.")
tf.flags.DEFINE_string("data_in_dir", "/scratch2/sniu/kitti/",
                       "data input dir")
tf.flags.DEFINE_string("date", "2011_09_26",
                       "which date to use")
tf.flags.DEFINE_string("weights", "model_490.ckpt",
                       "which date to use")

# partition
tf.flags.DEFINE_string("partition_mode", 'grid', 
                       "currently only support regular grid partition")
tf.flags.DEFINE_boolean("fb_split", False, 
                       "if split data as foreground and background")
tf.flags.DEFINE_string("PW", '32', # x: width, y: length, coarse grain ,...., fine grain
                       "number of partitions for wideth, if it is list, it becomes the multi-scale") 
tf.flags.DEFINE_string("PL", '32', 
                       "number of partition for length, same as PW")
tf.flags.DEFINE_string("cell_max_points", '512', 
                       "maximum number of points in the cell")
tf.flags.DEFINE_string("cell_min_points", '50', 
                       "minimum number of points in the cell")
# tf.flags.DEFINE_string("cluster_mode", 'kmeans',
#                        "kmeans, spectral, dirichlet")

# compression
tf.flags.DEFINE_string("compression_mode", "pointnet",
                       "[kmeans|octree|pointnet|pointgrid], pointnet and pointgrid are neural network based")
tf.flags.DEFINE_string("loss_type", "chamfer",
                       "loss type [chamfer|emd]")
tf.flags.DEFINE_string("encoder", "pointnet",
                       "encoder configuration [pointnet]")
tf.flags.DEFINE_string("decoder", "pointnet",
                       "decoder configuration [pointnet|pointgrid]")
tf.flags.DEFINE_integer("num_points", 200,
                       "number of points in point cloud")
tf.flags.DEFINE_integer("batch_size", 256,
                       "batch size")
tf.flags.DEFINE_integer("latent_dim", 16,
                       "latent code dimension")
tf.flags.DEFINE_integer("top_k", 10,
                       "number of sampling points in sampling mode")

# training config
tf.flags.DEFINE_float("seed", 3122018, "Random seeds for results reproduction")
tf.flags.DEFINE_integer("num_epochs", 500,
                       "number of training epoch")
tf.flags.DEFINE_integer("decay_step", 200000,
                       "Decay step for lr decay [default: 200000]")
tf.flags.DEFINE_integer("save_freq", 20,
                      "learning rate")
tf.flags.DEFINE_float("learning_rate", 0.001,
                      "learning rate")




# drives = ['0001', '0002', '0005', '0011']
drives = ['0001']

# if FLAGS.partition_mode == 0:
#     train_dir = tf_util.create_dir(osp.join(FLAGS.top_out_dir, "f_%s_%s" % (FLAGS.compression_mode, FLAGS.loss)))
# elif FLAGS.partition_mode == 1:
#      train_dir = tf_util.create_dir(osp.join(FLAGS.top_out_dir, "b_%s_%s" % (FLAGS.compression_mode, FLAGS.loss)))
# elif FLAGS.partition_mode == 2:
#      train_dir = tf_util.create_dir(osp.join(FLAGS.top_out_dir, "%s_%s" % (FLAGS.compression_mode, FLAGS.loss)))
# elif FLAGS.partition_mode == 3:
#      train_dir = tf_util.create_dir(osp.join(FLAGS.top_out_dir, "%s_%s_%s" % (FLAGS.cluster_mode, FLAGS.compression_mode, FLAGS.loss)))

def train():
    util.create_dir(FLAGS.save_path)

    # step 1 Partition
    point_cell = util.LoadData(args=FLAGS, drives=drives)

    if FLAGS.fb_split == True:
      # step 2 compression
      model_f = AutoEncoder(FLAGS)
      model_f.train(point_cell, mode='foreground')

      model_d = AutoEncoder(FLAGS)
      model_d.train(point_cell, mode='background')

    else:
      model = AutoEncoder(FLAGS)
      model.train(point_cell)


    # step 3 merge
    # merge()

    # # step 4 evaluate
    # evaluation()

def test():

    point_cell = util.LoadData(args=FLAGS, drives=drives)
    sample_points, sample_info = point_cell.sample_points, point_cell.sample_info
    train_points, train_info = point_cell.train_points, point_cell.train_info

    origin_points_sample = point_cell.test_sweep[point_cell.sample_idx]
    origin_points_train = point_cell.cleaned_velo[point_cell.sample_idx_train]

    model = AutoEncoder(FLAGS)
    ckpt_name = FLAGS.weights
    sample_generated = model.predict(sample_points, sample_info, ckpt_name)
    train_generated = model.predict(train_points, train_info, ckpt_name)

    # measure emd distance
    # x_reconstr, gt = tf.placeholder(tf.float32, [1, None, 3]), tf.placeholder(tf.float32, [1, None, 3])
    # match = approx_match(x_reconstr, gt)
    # emd_loss = tf.reduce_mean(match_cost(x_reconstr, gt, match))
    # emd_max = tf.reduce_max(match_cost(x_reconstr, gt, match))
    # emd_min = tf.reduce_min(match_cost(x_reconstr, gt, match))

    # config = tf.ConfigProto(
    #     device_count = {'GPU': 0}
    # )
    # sess = tf.Session(config=config)
    # feed_dict = {x_reconstr: np.expand_dims(reconstruction, 0), 
    #              gt: np.expand_dims(origin_points, 0)}
    # loss_max, loss_min, loss = sess.run([emd_max, emd_min, emd_loss], feed_dict)
    # print ("emd max loss: {}, emd min loss: {}, emd loss: {}".format(loss_max, loss_min, loss))
    # # print ("emd loss: {}".format(sess.run([emd_loss], feed_dict)))

    reconstruction_sample = point_cell.reconstruct_scene(sample_generated, sample_info)
    util.visualize_3d_points(origin_points_sample, dir='.', filename='orig_sample')
    util.visualize_3d_points(reconstruction_sample, dir='.', filename='recon_sample')

    reconstruction_train = point_cell.reconstruct_scene(train_generated, train_info)
    util.visualize_3d_points(origin_points_train, dir='.', filename='orig_train')
    util.visualize_3d_points(reconstruction_train, dir='.', filename='recon_train')

if __name__ == '__main__':
    # if is_training == True:
    train()
    # else:
        # test()