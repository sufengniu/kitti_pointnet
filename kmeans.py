
import numpy as np
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans

import tensorflow as tf
from external.structural_losses.tf_nndistance import nn_distance
from external.structural_losses.tf_approxmatch import approx_match, match_cost


all_pc_data, all_meta_data, _ = data.load_pc_data(batch_size)

num_anchors_2 = 34

# MLP: all_points [batch, 200, 2] -> MLP -> node_feature [batch, 200, 10]
gt = tf.placeholder(tf.float32, [batch_size, origin_num_points, 3])
meta = tf.placeholder(tf.int32, [batch_size])
is_training = tf.placeholder(tf.bool, shape=())


sample_index = tf.random_uniform([num_anchors_2], minval=0, maxval=num_points, dtype=tf.int32)
x_subsample = tf.gather(gt, sample_index, axis=1)
cost_p1_p2_rand, _, cost_p2_p1_rand, _ = nn_distance(x_subsample, gt)
sample_loss = tf.reduce_mean(cost_p1_p2_rand) + tf.reduce_mean(cost_p2_p1_rand)

kmeans_center = tf.placeholder(tf.float32, [batch_size, num_anchors_2, 3])
cost_p1_p2_kmeans, _, cost_p2_p1_kmeans, _ = nn_distance(kmeans_center, gt)
kmeans_loss = tf.reduce_mean(cost_p1_p2_kmeans) + tf.reduce_mean(cost_p2_p1_kmeans)

sess = tf.InteractiveSession()

loss_rand = []
loss_ae = []
loss_cluster = []
for j in range(50):
    # X = tf_util.generate_batch_data(batch_size, num_points)
    X, X_meta = all_pc_data[j*batch_size:(j+1)*batch_size], all_meta_data[j*batch_size:(j+1)*batch_size]

    center = []
    for i_batch in range(X.shape[0]):
        kmeans = KMeans(n_clusters=num_anchors_2, random_state=0).fit(X[i_batch,:,:])
        center.append(kmeans.cluster_centers_)
    center = np.stack(center)

    random_loss, cluster_loss, sample = sess.run([sample_loss, kmeans_loss, x_subsample], feed_dict={gt: X, kmeans_center: center, is_training: True})
    loss_rand.append(random_loss)
    loss_cluster.append(cluster_loss)
   
loss_rand = np.stack(loss_rand)
loss_cluster = np.stack(loss_cluster)
print loss_ae.mean()
print loss_rand.mean()
print loss_cluster.mean()