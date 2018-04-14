import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_util
import data
import os
import os.path as osp

from sklearn.cluster import KMeans

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
tf.flags.DEFINE_string("loss", "chamfer",
                       "loss type [chamfer|emd]")
tf.flags.DEFINE_integer("split_mode", 3,
                       "0: foreground, 1: background, 2: all points, 3: kmeans on foreground")
tf.flags.DEFINE_integer("num_points", 200,
                       "number of points in point cloud")
tf.flags.DEFINE_integer("batch_size", 128,
                       "batch size")
tf.flags.DEFINE_integer("latent_dim", 128,
                       "latent code dimension")
tf.flags.DEFINE_integer("top_k", 10,
                       "number of sampling points in sampling mode")
tf.flags.DEFINE_integer("num_epochs", 5000,
                       "number of training epoch")
tf.flags.DEFINE_integer("decay_step", 200000,
                       "Decay step for lr decay [default: 200000]")
tf.flags.DEFINE_float("learning_rate", 0.001,
                      "learning rate")


# Data setting
batch_size = FLAGS.batch_size
origin_num_points = 800 if FLAGS.split_mode==3 else FLAGS.num_points
k = FLAGS.top_k
latent_dim = FLAGS.latent_dim
# Tensorflow code below
n_filters = 10
lr = FLAGS.learning_rate # learning rate
num_epochs = FLAGS.num_epochs

DECAY_RATE = 0.8
DECAY_STEP = FLAGS.decay_step
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

if FLAGS.split_mode == 0:
    train_dir = osp.join(FLAGS.top_out_dir, "f_%s_%s_%s" % (FLAGS.encoder, FLAGS.decoder, FLAGS.loss))
elif FLAGS.split_mode == 1:
     train_dir = osp.join(FLAGS.top_out_dir, "b_%s_%s_%s" % (FLAGS.encoder, FLAGS.decoder, FLAGS.loss))
elif FLAGS.split_mode == 2:
     train_dir = osp.join(FLAGS.top_out_dir, "%s_%s_%s" % (FLAGS.encoder, FLAGS.decoder, FLAGS.loss))
elif FLAGS.split_mode == 3:
     train_dir = osp.join(FLAGS.top_out_dir, "kmeans_%s_%s_%s" % (FLAGS.encoder, FLAGS.decoder, FLAGS.loss))

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*batch_size,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def eval():
    # MLP: all_points [batch, 200, 2] -> MLP -> node_feature [batch, 200, 10]
    gt = tf.placeholder(tf.float32, [None, origin_num_points, 3])
    meta = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.bool, shape=())

    batch = tf.get_variable('batch', [],
        initializer=tf.constant_initializer(0), trainable=False)
    bn_decay = get_bn_decay(batch)

    mask = tf.sequence_mask(meta, maxlen=origin_num_points, dtype=tf.float32)
    mask = tf.expand_dims(mask, -1)

    if FLAGS.encoder == "edge":
        input_image = gt
        adj_matrix = tf_util.pairwise_distance(input_image)
        nn_idx = tf_util.knn(adj_matrix, k=k)

        # ------- encoder  -------   
        # two-layer conv
        edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)
        net = tf_util.neighbor_conv(edge_feature, "encoder_1", is_training, n_filters=64, bn_decay=bn_decay)

        # adj_matrix = tf_util.pairwise_distance(net)
        # nn_idx = tf_util.knn(adj_matrix, k=k)
        edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)
        net = tf_util.neighbor_conv(edge_feature, "encoder_2", is_training, n_filters=latent_dim, bn_decay=bn_decay)
    elif FLAGS.encoder == "point":
        net = gt
        net = tf_util.point_conv(net, "encoder_1", is_training, n_filters=64, bn_decay=bn_decay)
        net = tf_util.point_conv(net, "encoder_2", is_training, n_filters=latent_dim, bn_decay=bn_decay)

    net = net * mask

    # max pool
    code = tf.reduce_max(net, axis=-2, keepdims=False)
    print (code)

    # ------- decoder  ------- 
    if FLAGS.decoder == "fc":
        x_reconstr = tf_util.fully_connected_decoder(code, origin_num_points, is_training, bn_decay=bn_decay)
    elif FLAGS.decoder == "grid":
        code = tf.expand_dims(code, -2)
        map_size = [20, 10]
        global_code_tile = tf.tile(code, [1, origin_num_points, 1])
        local_code = tf_util.fold(map_size[0], map_size[1])
        local_code = tf.reshape(local_code, [1, origin_num_points, 2])
        local_code_tile = tf.tile(local_code, [batch_size, 1, 1])
        all_code = tf.concat([local_code_tile, global_code_tile], axis = -1)

        net = tf_util.point_conv(all_code, "decoder_1", is_training, 100, activation_function=tf.nn.relu, bn_decay=bn_decay)
        net = tf_util.point_conv(net, "decoder_2", is_training, 100, activation_function=tf.nn.relu, bn_decay=bn_decay)
        net = tf_util.point_conv(net, "decoder_3", is_training, 50, activation_function = tf.nn.relu, bn_decay=bn_decay)
        net = tf_util.point_conv(net, "decoder_4", is_training, 10, activation_function = tf.nn.relu, bn_decay=bn_decay)
        x_reconstr = tf_util.point_conv(net, "decoder_5", is_training, 3, activation_function = tf.nn.tanh, bn_decay=bn_decay)

    x_reconstr = x_reconstr * mask
    #  -------  loss + optimization  ------- 
    if FLAGS.loss == "emd":
        match = approx_match(x_reconstr, gt)
        reconstruction_loss = tf.reduce_mean(match_cost(x_reconstr, gt, match))
    elif FLAGS.loss == "chamfer":
        cost_p1_p2, _, cost_p2_p1, _ = nn_distance(x_reconstr, gt)
        reconstruction_loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)

    # reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    saver.restore(sess, train_dir + '/model_%s.ckpt' % (str(180)))

    all_pc_data, all_meta_data, _ = data.load_pc_data(batch_size, train=True, split_mode=FLAGS.split_mode)

    record_loss = []
    record_mean, record_var, record_mse = [], [], []

    for j in range(all_pc_data.shape[0]//batch_size):
        X_pc, X_meta = all_pc_data[j*batch_size:(j+1)*batch_size], all_meta_data[j*batch_size:(j+1)*batch_size]
        emd_loss, recon = sess.run([reconstruction_loss, x_reconstr], feed_dict={gt: X_pc, meta: X_meta, is_training: False})
        record_loss.append(emd_loss)
        print ("loss: {}".format(emd_loss))
        mean, var, mse = tf_util.dist_stat(recon, X_pc, X_meta)
        record_mean.append(mean)
        record_var.append(var)
        record_mse.append(mse)
    print (np.array(record_loss).mean())
    print (np.array(record_mean).mean(), np.array(record_var).mean(), np.array(record_mse).mean())

    tf_util.dist_vis(recon, X_pc, X_meta)



def eval_baseline():
    # MLP: all_points [batch, 200, 2] -> MLP -> node_feature [batch, 200, 10]
    gt = tf.placeholder(tf.float32, [None, origin_num_points, 3])
    meta = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.bool, shape=())

    mask = tf.sequence_mask(meta, maxlen=origin_num_points, dtype=tf.float32)
    mask = tf.expand_dims(mask, -1)


    #  -------  loss + optimization  ------- 
    num_anchors = 43
    sample_index = tf.random_uniform([num_anchors], minval=0, maxval=origin_num_points, dtype=tf.int32)
    x_subsample = tf.gather(gt, sample_index, axis=1)
    cost_p1_p2_rand, _, cost_p2_p1_rand, _ = nn_distance(x_subsample, gt)
    sample_loss = tf.reduce_mean(cost_p1_p2_rand) + tf.reduce_mean(cost_p2_p1_rand)

    kmeans_center = tf.placeholder(tf.float32, [batch_size, num_anchors, 3])
    cost_p1_p2_kmeans, _, cost_p2_p1_kmeans, _ = nn_distance(kmeans_center, gt)
    kmeans_loss = tf.reduce_mean(cost_p1_p2_kmeans) + tf.reduce_mean(cost_p2_p1_kmeans)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    all_pc_data, all_meta_data, _ = data.load_pc_data(batch_size, train=True)


    # kmeans
    record_sample_loss, record_cluster_loss = [], []
    record_mean_cluster, record_var_cluster, record_mse_cluster = [], [], []
    record_mean_random, record_var_random, record_mse_random = [], [], []
    # for j in range(all_pc_data.shape[0]//batch_size):
    for j in range(100):
        X_pc, X_meta = all_pc_data[j*batch_size:(j+1)*batch_size], all_meta_data[j*batch_size:(j+1)*batch_size]
        center = []
        for i_batch in range(X_pc.shape[0]):
            kmeans = KMeans(n_clusters=num_anchors, random_state=0).fit(X_pc[i_batch,:,:])
            center.append(kmeans.cluster_centers_)
        center = np.stack(center)
        random_loss, cluster_loss, recon_sample = sess.run([sample_loss, kmeans_loss, x_subsample], feed_dict={gt: X_pc, kmeans_center: center, is_training: False})
        record_cluster_loss.append(cluster_loss)
        record_sample_loss.append(random_loss)
        mean_cluster, var_cluster, mse_cluster = tf_util.dist_stat(center, X_pc, X_meta)
        mean_random, var_random, mse_random = tf_util.dist_stat(recon_sample, X_pc, X_meta)
        record_mean_cluster.append(mean_cluster)
        record_var_cluster.append(var_cluster)
        record_mse_cluster.append(mse_cluster)
        record_mean_random.append(mean_random)
        record_var_random.append(var_random)
        record_mse_random.append(mse_random)
        
    print (np.array(record_cluster_loss).mean())
    print (np.array(record_sample_loss).mean())
    print (np.array(record_mean_cluster).mean(), np.array(record_var_cluster).mean(), np.array(record_mse_cluster).mean())
    print (np.array(record_mean_random).mean(), np.array(record_var_random).mean(), np.array(record_mse_random).mean())



# eval()
eval_baseline()