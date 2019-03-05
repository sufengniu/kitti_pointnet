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
from model import AutoEncoder, StackAutoEncoder
import tf_util
import util

from external.structural_losses.tf_nndistance import nn_distance
from external.structural_losses.tf_approxmatch import approx_match, match_cost

FLAGS = tf.flags.FLAGS

# Training options.
tf.flags.DEFINE_string("save_path", "/scratch1/sniu/pc/test/",
                       "Checkpointing directory.")
tf.flags.DEFINE_string("result_output_path", None,
                       "Checkpointing directory.")
tf.flags.DEFINE_string("data_in_dir", "/scratch2/sniu/kitti/",
                       "data input dir")
tf.flags.DEFINE_string("hdmap_dir", "/scratch2/sniu/hdmap/",
                       "data input dir")
tf.flags.DEFINE_string("date", "2011_09_26",
                       "which date to use")
tf.flags.DEFINE_string("weights", "model-260.ckpt",
                       "which checkpoint to use")
tf.flags.DEFINE_string("dataset", "kitti",
                       "which data to use [kitti|hdmap]")
tf.flags.DEFINE_boolean("load_prestored_data", False, 
                       "default setting is to compute online")
tf.flags.DEFINE_string("checkpoint_path", None, 
                       "a path that stores a pretrained model")
tf.flags.DEFINE_string("checkpoint_number", None, 
                       "number of epoch")

# partition
tf.flags.DEFINE_string("partition_mode", 'grid', 
                       "how to partition the sweep, grid partition, or range view partition [grid|range]")
tf.flags.DEFINE_boolean("fb_split", False, 
                       "if split data as foreground and background")
tf.flags.DEFINE_string("PW", '32,16', # x: width, y: length, 'fine grain ,...., coarse grain'
                       "number of partitions for wideth, if it is list, it becomes the multi-scale") 
tf.flags.DEFINE_string("PL", '32,16', 
                       "number of partition for length, same as PW")
tf.flags.DEFINE_string("cell_max_points", '512,512', 
                       "maximum number of points in the cell")
tf.flags.DEFINE_string("cell_min_points", '50,50', 
                       "minimum number of points in the cell")
# tf.flags.DEFINE_string("cluster_mode", 'kmeans',
#                        "kmeans, spectral, dirichlet")

# compression
tf.flags.DEFINE_string("mode", "autoencoder",
                       "[kmeans|random|octree|autoencoder|encoder|decoder], random, kmeans, octree or autoencoder based")
tf.flags.DEFINE_string("loss_type", "chamfer",
                       "loss type [chamfer|emd]")
tf.flags.DEFINE_string("encoder", "pointnet",
                       "encoder configuration [pointnet|magic|dgcnn|inception]")
tf.flags.DEFINE_string("decoder", "pointnet",
                       "decoder configuration [pointnet|pointgrid]")
tf.flags.DEFINE_string("pooling", "max",
                       "use mean pooling")
tf.flags.DEFINE_integer("batch_size", 512,
                       "batch size")
tf.flags.DEFINE_integer("latent_dim", 18,
                       "latent code dimension")
tf.flags.DEFINE_integer("top_k", 20,
                       "number of k nearest neighbor for dgcnn and inception")
tf.flags.DEFINE_boolean("rotation", False, 
                       "if adding rotation module or not")


# training config
tf.flags.DEFINE_integer("num_gpus", 2,
                       "number of gpu")
tf.flags.DEFINE_boolean("stacked", False, 
                       "if it is multi-scale or not")
tf.flags.DEFINE_boolean("training", False, 
                       "whether it is train or evaluation")
tf.flags.DEFINE_boolean("compress_all_kitti", False, 
                       "whether compress all kitti data or not, this is used for prediction")
tf.flags.DEFINE_float("seed", 3122018, "Random seeds for results reproduction")
tf.flags.DEFINE_integer("num_epochs", 280,
                       "number of training epoch")
tf.flags.DEFINE_integer("decay_step", 200000,
                       "Decay step for lr decay [default: 200000]")
tf.flags.DEFINE_integer("save_freq", 20,
                      "learning rate")
tf.flags.DEFINE_float("learning_rate", 0.001,
                      "learning rate")

train_drives = ['0001', '0002', '0005', '0011', '0013', '0014', '0015', '0017', '0018', '0019',
                '0020', '0022', '0023', '0027', '0028', '0029', '0032', '0035', '0036', '0039',
                '0046', '0048', '0051', '0052', '0056', '0057', '0059', '0060', '0061', '0064',
                '0070', '0079', '0084', '0052', '0056', '0057', '0059', '0060', '0061', '0064']
test_drives = ['0101', '0104', '0106', '0113', '0117']


def evaluate_sweep():
    util.create_dir(FLAGS.save_path)
    if FLAGS.result_output_path is not None:
        util.create_dir(FLAGS.result_output_path)

    # step 1 Partition
    point_cell = util.LoadData(args=FLAGS, train_drives=train_drives, test_drives=test_drives)

    level = len(list(map(int, FLAGS.PL.split(','))))
    print ('training grain from corase to fine')

    if level >= 2:
        model = StackAutoEncoder(FLAGS)
    else:
        model = AutoEncoder(FLAGS)
    ckpt_name = FLAGS.weights
    # ckpt_path = util.create_dir(osp.join(model.save_path, 'level_%s' % (model.current_level)))
    # model.saver.restore(model.sess, os.path.join(ckpt_path, ckpt_name))

    # model.train(point_cell)
    # ckpt_name = 'model-10.ckpt'
    if FLAGS.compress_all_kitti == True:
        model.compress(point_cell, ckpt_name)
    else:
        model.predict_test(point_cell, ckpt_name, FLAGS.mode)
        point_cell.test_compression_rate()
    
def restore_model_from_checkpoint(model):
    if FLAGS.checkpoint_path is not None and FLAGS.checkpoint_number is not None:
        checkpoint = FLAGS.checkpoint_path + 'model-' + str(FLAGS.checkpoint_number) + '.ckpt'
        model.saver.restore(model.sess, checkpoint)
        print("Restore the checkpoint at: {}".format(checkpoint))
    else:
        print("Training from scratch!")


def train():
    util.create_dir(FLAGS.save_path)

    # step 1 Partition
    point_cell = util.LoadData(args=FLAGS, train_drives=train_drives, test_drives=test_drives)

    level = len(list(map(int, FLAGS.PL.split(','))))
    print ('training grain from corase to fine')

    if level >= 2: # multi-scale autoencoder
        if FLAGS.fb_split == True:
            # step 2 compression
            model_f = StackAutoEncoder(FLAGS)
            model_f.train(point_cell, mode='foreground')
            # model_f.predict()
            
            model_d = StackAutoEncoder(FLAGS)
            model_d.train(point_cell, mode='background')

        else:
            model = StackAutoEncoder(FLAGS)
            restore_model_from_checkpoint(model)
            model.train(point_cell)

    else:
        l = 0
        if FLAGS.fb_split == True:
            # step 2 compression
            model_f = AutoEncoder(FLAGS)
            model_f.train(point_cell, mode='foreground')
            # model_f.predict()
            
            model_d = AutoEncoder(FLAGS)
            model_d.train(point_cell, mode='background')

        else:
            model = AutoEncoder(FLAGS)
            restore_model_from_checkpoint(model)
            model.train(point_cell)

    point_cell.test_compression_rate()

    ckpt_name = FLAGS.weights

def test():

    point_cell = util.LoadData(args=FLAGS, train_drives=train_drives, test_drives=test_drives)
    sample_points = point_cell.sample_points
    train_points, train_info = point_cell.train_points, point_cell.train_info

    origin_points_sample = point_cell.test_sweep[point_cell.sample_idx]
    origin_points_train = point_cell.cleaned_velo[point_cell.sample_idx_train]

    model = AutoEncoder(FLAGS)
    ckpt_name = FLAGS.weights
    sample_generated = model.predict(sample_points, sample_info, ckpt_name)
    train_generated = model.predict(train_points, train_info, ckpt_name)

    # plot
    i = 170
    p = np.random.choice(point_cell.train_cleaned_velo[i].shape[0], 30000, replace=False)
    points = point_cell.train_cleaned_velo[i][p]
    util.visualize_3d_points(points, filename='test')

    reconstruction_sample = point_cell.reconstruct_scene(sample_generated, sample_info)
    util.visualize_3d_points(origin_points_sample, dir='.', filename='orig_sample')
    util.visualize_3d_points(reconstruction_sample, dir='.', filename='recon_sample')

    reconstruction_train = point_cell.reconstruct_scene(train_generated, train_info)
    util.visualize_3d_points(origin_points_train, dir='.', filename='orig_train')
    util.visualize_3d_points(reconstruction_train, dir='.', filename='recon_train')

if __name__ == '__main__':
    if FLAGS.training == True:
        train()
    else:
        evaluate_sweep()
