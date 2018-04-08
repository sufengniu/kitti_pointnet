import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_util
import data

from external.structural_losses.tf_nndistance import nn_distance
from external.structural_losses.tf_approxmatch import approx_match, match_cost



# Data setting
batch_size = 32
origin_num_points = 200
k = 10
# Tensorflow code below
n_filters = 10
lr = 0.001 # learning rate
num_epochs = 3000


# MLP: all_points [batch, 200, 2] -> MLP -> node_feature [batch, 200, 10]
gt = tf.placeholder(tf.float32, [None, origin_num_points, 3])
meta = tf.placeholder(tf.int32, [None])
is_training = tf.placeholder(tf.bool, shape=())

mask = tf.sequence_mask(meta, maxlen=origin_num_points, dtype=tf.float32)
mask = tf.expand_dims(mask, -1)

# downsample points from gt
# sample_index = tf.random_uniform([num_points], minval=0, maxval=origin_num_points, dtype=tf.int32)
# input_image = tf.gather(gt, sample_index, axis=1)
input_image = gt
adj_matrix = tf_util.pairwise_distance(input_image)
nn_idx = tf_util.knn(adj_matrix, k=k)

# ------- encoder  -------   
# two-layer conv
edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)
net = tf_util.neighbor_conv(edge_feature, "encoder_1", is_training, n_filters = 20)

# adj_matrix = tf_util.pairwise_distance(net)
# nn_idx = tf_util.knn(adj_matrix, k=k)
edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)
net = tf_util.neighbor_conv(edge_feature, "encoder_2", is_training, n_filters = 50)

net = net * mask

# max pool
code = tf.reduce_max(net, axis=-2, keepdims=True)
print (code)

# ------- decoder  ------- 
# x_reconstr = tf_util.fully_connected_decoder(code, origin_num_points, is_training)

map_size = [20, 10]
global_code_tile = tf.tile(code, [1, origin_num_points, 1])
local_code = tf_util.fold(map_size[0], map_size[1])
local_code = tf.reshape(local_code, [1, origin_num_points, 2])
local_code_tile = tf.tile(local_code, [batch_size, 1, 1])
all_code = tf.concat([local_code_tile, global_code_tile], axis = -1)

net = tf_util.point_conv(all_code, "1", is_training, 100, activation_function = tf.nn.relu)
net = tf_util.point_conv(net, "2", is_training, 100, activation_function = tf.nn.relu)
net = tf_util.point_conv(net, "3", is_training, 50, activation_function = tf.nn.relu)
net = tf_util.point_conv(net, "4", is_training, 10, activation_function = tf.nn.relu)
x_reconstr = tf_util.point_conv(net, "5", is_training, 3, activation_function = tf.nn.tanh)

x_reconstr = x_reconstr * mask
#  -------  loss + optimization  ------- 
# match = approx_match(x_reconstr, gt)
# reconstruction_loss = tf.reduce_mean(match_cost(x_reconstr, gt, match))
cost_p1_p2, _, cost_p2_p1, _ = nn_distance(x_reconstr, gt)
reconstruction_loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)

optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(reconstruction_loss)
sess = tf.InteractiveSession()
saver = tf.train.Saver()
tf.global_variables_initializer().run()

all_pc_data, all_meta_data, _ = data.load_pc_data(batch_size)

record_loss = []
for i in range(num_epochs):
    for j in range(all_pc_data.shape[0]//batch_size):
        X_pc, X_meta = all_pc_data[j*batch_size:(j+1)*batch_size], all_meta_data[j*batch_size:(j+1)*batch_size]
        _, emd_loss, recon = sess.run([train_op, reconstruction_loss, x_reconstr], feed_dict={gt: X_pc, meta: X_meta, is_training: True})
        record_loss.append(emd_loss)

    data.shuffle_data(all_pc_data, all_meta_data)
    print ("iteration: {}, loss: {}".format(i, emd_loss))

tf_util.dist_vis(recon, X_pc, X_meta)

saver.save(sess, 'model/grid_model_chamfer.ckpt')


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