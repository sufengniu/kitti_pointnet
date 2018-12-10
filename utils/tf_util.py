import numpy as np
import tensorflow as tf
import os
import os.path as osp
import pandas as pd
import scipy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, load_all_point_clouds_under_folder


def _variable_on_cpu(name, shape, initializer, use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer
  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 1D convolution with non-linear operation.
  Args:
    inputs: 3-D tensor variable BxLxC
    num_output_channels: int
    kernel_size: int
    scope: string
    stride: int
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_size,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    outputs = tf.nn.conv1d(inputs, kernel,
                           stride=stride,
                           padding=padding)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)

    if bn:
      outputs = batch_norm_for_conv1d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs




def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 2D convolution with non-linear operation.
  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training, scope='bn')

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs


def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=0.0,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None):
  """ 2D convolution transpose with non-linear operation.
  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable
  Returns:
    Variable tensor
  Note: conv2d(conv2d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_output_channels, num_in_channels] # reversed to conv2d
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      
      # from slim.convolution2d_transpose
      def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
          dim_size *= stride_size

          if padding == 'VALID' and dim_size is not None:
            dim_size += max(kernel_size - stride_size, 0)
          return dim_size

      # caculate output shape
      batch_size = inputs.get_shape()[0].value
      height = inputs.get_shape()[1].value
      width = inputs.get_shape()[2].value
      out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
      out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
      output_shape = [batch_size, out_height, out_width, num_output_channels]

      outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training, scope='bn')

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs

   

def conv3d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 3D convolution with non-linear operation.
  Args:
    inputs: 5-D tensor variable BxDxHxWxC
    num_output_channels: int
    kernel_size: a list of 3 ints
    scope: string
    stride: a list of 3 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_d, kernel_h, kernel_w,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.conv3d(inputs, kernel,
                           [1, stride_d, stride_h, stride_w, 1],
                           padding=padding)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
    
    if bn:
      outputs = batch_norm_for_conv3d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs

def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    is_training=None):
  """ Fully connected layer with non-linear operation.
  
  Args:
    inputs: 2-D tensor BxN
    num_outputs: int
  
  Returns:
    Variable tensor of size B x num_outputs.
  """
  with tf.variable_scope(scope) as sc:
    num_input_units = inputs.get_shape()[-1].value

    weights = _variable_with_weight_decay('weights',
                                          shape=[num_input_units, num_outputs],
                                          use_xavier=use_xavier,
                                          stddev=stddev,
                                          wd=weight_decay)
    outputs = tf.matmul(inputs, weights)
    biases = _variable_on_cpu('biases', [num_outputs],
                             tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
     
    if bn:
      outputs = batch_norm_for_fc(outputs, is_training, 'bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D max pooling.
  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs

def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D avg pooling.
  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.avg_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs


def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D max pooling.
  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.max_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs

def avg_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D avg pooling.
  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.avg_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs


def batch_norm_template(inputs, is_training, scope, moments_dims_unused, data_format='NHWC'):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
      data_format:   'NHWC' or 'NCHW'
  Return:
      normed:        batch-normalized maps
  """
  return tf.layers.batch_normalization(inputs, training=is_training)


# def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
#   """ Batch normalization on convolutional maps and beyond...
#   Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
#   Args:
#       inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
#       is_training:   boolean tf.Varialbe, true indicates training phase
#       scope:         string, variable scope
#       moments_dims:  a list of ints, indicating dimensions for moments calculation
#       bn_decay:      float or float tensor variable, controling moving average weight
#   Return:
#       normed:        batch-normalized maps
#   """
#   with tf.variable_scope(scope) as sc:
#     num_channels = inputs.get_shape()[-1].value
#     beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
#                        name='beta', trainable=True)
#     gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
#                         name='gamma', trainable=True)
#     batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
#     decay = bn_decay if bn_decay is not None else 0.9
#     ema = tf.train.ExponentialMovingAverage(decay=decay)
#     # Operator that maintains moving averages of variables.
#     ema_apply_op = tf.cond(is_training,
#                            lambda: ema.apply([batch_mean, batch_var]),
#                            lambda: tf.no_op())
    
#     # Update moving average and return current batch's avg and var.
#     def mean_var_with_update():
#       with tf.control_dependencies([ema_apply_op]):
#         return tf.identity(batch_mean), tf.identity(batch_var)
    
#     # ema.average returns the Variable holding the average of var.
#     mean, var = tf.cond(is_training,
#                         mean_var_with_update,
#                         lambda: (ema.average(batch_mean), ema.average(batch_var)))
#     normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
#   return normed


def batch_norm_for_fc(inputs, is_training, scope):
  """ Batch normalization on FC data.
  
  Args:
      inputs:      Tensor, 2D BxC input
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,])


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 1D convolutional maps.
  
  Args:
      inputs:      Tensor, 3D BLC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1])



  
def batch_norm_for_conv2d(inputs, is_training, scope):
  """ Batch normalization on 2D convolutional maps.
  
  Args:
      inputs:      Tensor, 4D BHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2])



def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 3D convolutional maps.
  
  Args:
      inputs:      Tensor, 5D BDHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2,3])


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
  """ Dropout layer.
  Args:
    inputs: tensor
    is_training: boolean tf.Variable
    scope: string
    keep_prob: float in [0,1]
    noise_shape: list of ints
  Returns:
    tensor variable
  """
  with tf.variable_scope(scope) as sc:
    outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
    return outputs


def pairwise_distance(point_cloud):
  """Compute pairwise distance of a point cloud.

  Args:
    point_cloud: tensor (batch_size, num_points, num_dims)

  Returns:
    pairwise distance: (batch_size, num_points, num_points)
  """
  og_batch_size = point_cloud.get_shape().as_list()[0]
  point_cloud = tf.squeeze(point_cloud)
  if og_batch_size == 1:
    point_cloud = tf.expand_dims(point_cloud, 0)
    
  point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
  point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
  point_cloud_inner = -2*point_cloud_inner
  point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
  point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
  return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


def knn(adj_matrix, k=20):
  """Get KNN based on the pairwise distance.
  Args:
    pairwise distance: (batch_size, num_points, num_points)
    k: int

  Returns:
    nearest neighbors: (batch_size, num_points, k)
  """
  neg_adj = -adj_matrix
  _, nn_idx = tf.nn.top_k(neg_adj, k=k)
  return nn_idx


def get_edge_feature(point_cloud, nn_idx, k=20):
  """Construct edge feature for each point
  Args:
    point_cloud: (batch_size, num_points, 1, num_dims)
    nn_idx: (batch_size, num_points, k)
    k: int

  Returns:
    edge features: (batch_size, num_points, k, num_dims)
  """
  # og_batch_size = point_cloud.get_shape().as_list()[0]
  # point_cloud = tf.squeeze(point_cloud)
  # if og_batch_size == 1:
  #   point_cloud = tf.expand_dims(point_cloud, 0)

  point_cloud_central = point_cloud

  point_cloud_shape = point_cloud.get_shape()
  batch_size = tf.shape(point_cloud)[0]
  num_points = point_cloud_shape[1].value
  num_dims = point_cloud_shape[2].value

  idx_ = tf.range(batch_size) * num_points
  idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 

  point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
  point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
  point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

  point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])
  
  edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors], axis=-1)
  edge_feature = tf.reshape(edge_feature, [-1, num_points, k, 2*num_dims])

  return edge_feature


def neighbor_conv(input_image, name, is_training, n_filters=10, activation_function=tf.nn.relu, bn_decay=None):
   net = conv2d(input_image, n_filters, [1,1],
                padding='VALID', stride=[1,1],
                activation_fn=activation_function,
                bn=True, is_training=is_training,
                scope=name, bn_decay=bn_decay)
   net = tf.reduce_max(net, axis=-2, keepdims=False)
   net = tf.nn.relu(net)
   # net = tf.squeeze(net, -2)

   return net

def fully_connected_decoder(net, num_points, is_training):

   # batch_size = tf.shape(net)[0]

   num_code = net.get_shape()[1]
   net = tf.reshape(net, [-1, num_code])

   net = fully_connected(net, 256, scope='fc1', bn=True,
                      is_training=is_training,
                      activation_fn=tf.nn.relu)
   net = fully_connected(net, 256, scope='fc2', bn=True,   
                      is_training=is_training,
                      activation_fn=tf.nn.relu)
   net = fully_connected(net, num_points*3, scope='fc3', bn=True,  
                      is_training=is_training,
                      activation_fn=tf.nn.tanh)

   net = tf.reshape(net, [-1, num_points, 3])

   return net

def point_conv(input_image, is_training, n_filters=[10], activation_function=tf.nn.relu, name=None):
   
   net = tf.expand_dims(input_image, -2)
   for idx, n_filter in enumerate(n_filters):
     net = conv2d(net, n_filter, [1,1],
                  padding='VALID', stride=[1,1],
                  activation_fn=activation_function,
                  bn=True, is_training=is_training,
                  scope=name+'_'+str(idx))
   net = tf.squeeze(net, -2)

   return net


def generate_bar(num_points):
   A = 0.5
   B = np.pi*np.random.uniform(0, 1)
   x = np.random.uniform(0, 1, [num_points])
   y = -B * np.square(x-A) + B * np.square(A)
   z = np.zeros_like(x)
   points = np.stack([x, y, z], axis=-1)
   return points

def rotation(points, theta):
   x = np.cos(theta)*points[:,0] - np.sin(theta)*points[:,1]
   y = np.sin(theta)*points[:,0] + np.cos(theta)*points[:,1]
   z = points[:,2]
   rotated_points = np.stack([x,y,z],axis=-1)
   return rotated_points

def generate_multiple_bars(num_points, bar_num=4):

   for i_bar in range(bar_num):

       theta = 2*np.pi*np.random.uniform(0, 1) # random_number
       points = rotation(generate_bar(num_points), theta)
       
       if 'all_points' in dir():
           all_points = np.concatenate([all_points, points], axis=0)
       else:
           all_points = points
   return all_points

def generate_batch_data(batch_size, num_points, bar_num=4):
   num_points_per_bar = int(num_points / bar_num)

   all_points = []
   for i_batch in range(batch_size):
       points = generate_multiple_bars(num_points_per_bar, bar_num)
       all_points.append(points)
   all_points = np.stack(all_points)

   return all_points


# def generate_pc_data(batch_size):
#   class_name = 'chair'
#   top_in_dir = "/scratch2/sniu/point_cloud/shape_net_core_uniform_samples_2048/"

#   syn_id = snc_category_to_synth_id()[class_name]
#   class_dir = osp.join(top_in_dir, syn_id)
#   all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)

#   train_batch, _, = all_pc_data.next_batch(batch_size)

#   return train_batch

# def generate_test_data(batch_size):
#   class_name = 'chair'
#   top_in_dir = "/scratch2/sniu/point_cloud/shape_net_core_uniform_samples_2048/"

#   syn_id = snc_category_to_synth_id()[class_name]
#   class_dir = osp.join(top_in_dir, syn_id)
#   all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)

#   feed_pc_test, _ = all_pc_data.full_test_data()

#   return feed_pc_test


def gmm_decoder(net, is_training, batch_size, num_points, num_anchors, n_filters, bn_decay=None):

   tile_num = int(num_points/num_anchors)

   # fully connected
   net = tf.reshape(net, [batch_size, -1])
   net = fully_connected(net, 256, scope='fc1', bn=True,
                      bn_decay=bn_decay, is_training=is_training,
                      activation_fn=tf.nn.relu)

   net = fully_connected(net, 256, scope='fc2', bn=True,   
                      bn_decay=bn_decay, is_training=is_training,
                      activation_fn=tf.nn.relu)

   net = fully_connected(net, num_anchors*n_filters, scope='fc3', bn=True,  
                      bn_decay=bn_decay, is_training=is_training,
                      activation_fn=tf.nn.tanh)
   net = tf.reshape(net, [-1, num_anchors, n_filters])

   # GMM
   e = tf.random_normal((batch_size, num_points, n_filters), mean=0.0, stddev=1.0)
   mu = tf.layers.dense(inputs=net, units=n_filters, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
   sigma = tf.nn.softplus(tf.layers.dense(inputs=net, units=n_filters, activation=None, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer()))+1e-6
   layer = tf.tile(mu, [1, tile_num, 1]) + tf.tile(sigma, [1, tile_num, 1]) * e
   layer = tf.expand_dims(layer, axis = -2)

   # postprocessing
   net = conv2d(layer, 3, [1,1],
                padding='VALID', stride=[1,1],
                activation_fn=tf.nn.tanh,
                bn=True, is_training=is_training,
                scope='1', bn_decay=bn_decay)   
   net = tf.squeeze(net, axis = -2)

   mu_net = conv2d(tf.expand_dims(mu, -2), 3, [1,1],
                padding='VALID', stride=[1,1],
                activation_fn=tf.nn.tanh,
                bn=True, is_training=is_training,
                scope='1', bn_decay=bn_decay) 
   mu_net = tf.squeeze(mu_net, axis = -2)
   return net, mu_net


def anchor_decoder(net, is_training, batch_size, num_points, num_anchors, n_filters, bn_decay=None):

    tile_num = int(num_points/num_anchors)

    # fully connected
    net = tf.reshape(net, [batch_size, -1])
    net = fully_connected(net, 256, scope='fc1', bn=True,
                      bn_decay=bn_decay, is_training=is_training,
                      activation_fn=tf.nn.relu)

    net = fully_connected(net, 256, scope='fc2', bn=True,   
                      bn_decay=bn_decay, is_training=is_training,
                      activation_fn=tf.nn.relu)

    net = fully_connected(net, num_anchors*n_filters, scope='fc3', bn=True,  
                      bn_decay=bn_decay, is_training=is_training,
                      activation_fn=tf.nn.tanh)
    net = tf.reshape(net, [-1, num_anchors, n_filters])

    # GMM
    e = tf.random_normal((batch_size, num_points, n_filters), mean=0.0, stddev=1.0)
    mu = tf.layers.dense(inputs=net, units=n_filters, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
    sigma = tf.nn.softplus(tf.layers.dense(inputs=net, units=n_filters, activation=None, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer()))+1e-6
    layer = tf.tile(mu, [1, tile_num, 1]) + tf.tile(sigma, [1, tile_num, 1]) * e
    layer = tf.expand_dims(layer, axis = -2)

    return mu, None


def fold(m, n):
    idx_1 = tf.tile(tf.expand_dims(tf.range(n,dtype=tf.float32), -1), [1,m])
    idx_2 = tf.tile(tf.expand_dims(tf.range(m,dtype=tf.float32), 0), [n,1])
   
    idx_1 = tf.reshape(idx_1, [-1, 1])
    idx_2 = tf.reshape(idx_2, [-1, 1])
   
    return tf.concat([idx_1, idx_2], axis=-1)

# def dist_stat(recon, orig, meta, num_points_upper_threshold=200):
#     batch_size = recon.shape[0]
#     num_anchor = recon.shape[1]
#     num_points = orig.shape[1]

#     all_dist = np.empty([0])
#     if num_anchor == num_points:
#         for i in range(batch_size):
#             idx = np.arange(min(meta[i], num_points_upper_threshold))
#             dist_tmp = scipy.spatial.distance.cdist(orig[i, idx], recon[i, idx])
#             dist_vec = np.min(dist_tmp, 1) # [num_points]
#             all_dist = np.concatenate([all_dist, dist_vec])
#     elif num_anchor < num_points:
#         for i in range(batch_size):
#             idx = np.arange(min(meta[i], num_anchor))
#             idx_orig = np.arange(min(meta[i], num_points_upper_threshold))
#             dist_tmp = scipy.spatial.distance.cdist(orig[i, idx_orig], recon[i, idx])
#             dist_vec = np.min(dist_tmp, 1) # [num_points]
#             all_dist = np.concatenate([all_dist, dist_vec])

#     return np.mean(all_dist), np.var(all_dist), np.sum(all_dist**2)

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
