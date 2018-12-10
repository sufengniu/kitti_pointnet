import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import tf_util
import data
import os
import os.path as osp

from external.structural_losses.tf_nndistance import nn_distance
from external.structural_losses.tf_approxmatch import approx_match, match_cost

FLAGS = tf.flags.FLAGS

# Optimizer parameters.
tf.flags.DEFINE_float("seed", 3122018, "Random seeds for results reproduction")

# Training options.
tf.flags.DEFINE_string("top_out_dir", "/scratch1/sniu/point_cloud/checkpoints",
                       "Checkpointing directory.")
tf.flags.DEFINE_string("top_in_dir", "/scratch2/sniu/point_cloud/shape_net_core_uniform_samples_2048/",
                       "data input dir")
tf.flags.DEFINE_string("encoder", "point",
                       "encoder type [edge|point]")
tf.flags.DEFINE_string("decoder", "fc",
                       "decoder type [fc|grid]")
tf.flags.DEFINE_string("loss", "chamfer",
                       "loss type [chamfer|emd]")
tf.flags.DEFINE_integer("split_mode", 0,
                       "0: foreground, 1: background, 2: all points, 3: clustering on foreground")
tf.flags.DEFINE_string("cluster_mode", 'kmeans',
                       "kmeans, spectral, dirichlet")
tf.flags.DEFINE_integer("num_points", 200,
                       "number of points in point cloud")
tf.flags.DEFINE_integer("batch_size", 512,
                       "batch size")
tf.flags.DEFINE_integer("latent_dim", 21,
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
origin_num_points = 1024 if FLAGS.split_mode==3 else FLAGS.num_points
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
    train_dir = tf_util.create_dir(osp.join(FLAGS.top_out_dir, "f_%s_%s_%s" % (FLAGS.encoder, FLAGS.decoder, FLAGS.loss)))
elif FLAGS.split_mode == 1:
     train_dir = tf_util.create_dir(osp.join(FLAGS.top_out_dir, "b_%s_%s_%s" % (FLAGS.encoder, FLAGS.decoder, FLAGS.loss)))
elif FLAGS.split_mode == 2:
     train_dir = tf_util.create_dir(osp.join(FLAGS.top_out_dir, "%s_%s_%s" % (FLAGS.encoder, FLAGS.decoder, FLAGS.loss)))
elif FLAGS.split_mode == 3:
     train_dir = tf_util.create_dir(osp.join(FLAGS.top_out_dir, "%s_%s_%s_%s" % (FLAGS.cluster_mode, FLAGS.encoder, FLAGS.decoder, FLAGS.loss)))

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        FLAGS.learning_rate,  # Base learning rate.
                        batch * batch_size,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate    

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*batch_size,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
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
        def body(net_in):
            row_idx = tf.range(net_in[1])
            col_idx_tmp = tf.expand_dims(tf.zeros_like(row_idx), -1)
            col_idx = tf.concat([col_idx_tmp, col_idx_tmp+1, col_idx_tmp+2], -1)
            img = net_in[0]
            row_idx = tf.tile(tf.expand_dims(row_idx, -1), [1, 3])
            idx = tf.stack([row_idx, col_idx])
            idx = tf.transpose(idx, [1,2,0])
            return tf.gather_nd(img, idx)
        input_image = tf.map_fn(body, (gt, meta), tf.float32)

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
        net = tf_util.point_conv(net, "decoder_5", is_training, 3, activation_function = None, bn_decay=bn_decay)
        net = tf.reshape(net, [-1, origin_num_points, 3])
        net_xy = tf.nn.tanh(net[:,:,0:2])
        net_z = tf.expand_dims(net[:,:,2], -1)
        x_reconstr = tf.concat([net_xy, net_z], -1)

    x_reconstr = x_reconstr * mask
    #  -------  loss + optimization  ------- 
    if FLAGS.loss == "emd":
        match = approx_match(x_reconstr, gt)
        reconstruction_loss = tf.reduce_mean(match_cost(x_reconstr, gt, match))
    elif FLAGS.loss == "chamfer":
        cost_p1_p2, _, cost_p2_p1, _ = nn_distance(x_reconstr, gt)
        reconstruction_loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)

    # reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    learning_rate = get_learning_rate(batch)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(reconstruction_loss, global_step=batch)
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    all_pc_data, all_meta_data, _ = data.load_pc_data(batch_size, split_mode=FLAGS.split_mode, cluster_mode=FLAGS.cluster_mode)

    record_loss = []
    for i in range(num_epochs):
        record_mean, record_var, record_mse = [], [], []
        for j in range(all_pc_data.shape[0]//batch_size):
            X_pc, X_meta = all_pc_data[j*batch_size:(j+1)*batch_size], all_meta_data[j*batch_size:(j+1)*batch_size]
            _, emd_loss, recon = sess.run([train_op, reconstruction_loss, x_reconstr], feed_dict={gt: X_pc, meta: X_meta, is_training: True})
            record_loss.append(emd_loss)
            mean, var, mse = tf_util.dist_stat(recon, X_pc, X_meta)
            record_mean.append(mean)
            record_var.append(var)
            record_mse.append(mse)

        data.shuffle_data(all_pc_data, all_meta_data)

        if i % 10 == 0:
            print ("iteration: {}, loss: {}".format(i, emd_loss))
            print ("mean: %.6f, var: %.6f, mse: %.6f" % (np.array(record_mean).mean(), np.array(record_var).mean(), np.array(record_mse).mean()))
            saver.save(sess, train_dir + '/model_%s.ckpt' % (str(i)))
            
    tf_util.dist_vis(recon, X_pc, X_meta)
    data.visualize_recon_points(X_pc[0], recon[0], file_name='train_compare_%d' % FLAGS.split_mode)

train()


# for i in range(10):
#     pc_visual = recon
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(pc_visual[i,:,0], pc_visual[i,:,1], pc_visual[i,:,2], 'ro')
#     ax.scatter(X_pc[i,:,0], X_pc[i,:,1], X_pc[i,:,2], 'bo')
#     plt.savefig('imgs/gen_%d.png' % (i), dpi=80)


# record_loss = []
# for i in range(num_epochs):
#    # X = tf_util.generate_pc_data(batch_size) # load real dataset
#    X = tf_util.generate_batch_data(batch_size, origin_num_points)
#    _, emd_loss, recon, mu_recon = sess.run([train_op, reconstruction_loss, x_reconstr, mu_reconstr], feed_dict={gt: X, is_training: True})
#    record_loss.append(emd_loss)
#    if i % 10 == 0:
#      print ("iteration: {}, loss: {}".format(i, emd_loss))

# fig = plt.figure()
# plt.plot(X[0,:,0], X[0,:,1], 'bs', mu_recon[0,:,0], mu_recon[0,:,1], 'ro')
# plt.savefig('orig')
# plt.close()

# fig = plt.figure()
# plt.plot(recon[0,:,0], recon[0,:,1], 'bs', mu_recon[0,:,0], mu_recon[0,:,1], 'ro')
# plt.savefig('recon')
# plt.close()