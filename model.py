from __future__ import print_function
import numpy as np
import tensorflow as tf
import util
import os
import os.path as osp
import sys
import collections

from sklearn.cluster import KMeans

from tqdm import *
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util
import provider
from transform_nets import *
import octree 
import h5py

from external.structural_losses.tf_nndistance import nn_distance
from external.structural_losses.tf_approxmatch import approx_match, match_cost
import time

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    From tensorflow tutorial: cifar10/cifar10_multi_gpu_train.py
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        #for g, _ in grad_and_vars:
        for g, v in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


class AutoEncoder():
    def __init__(self, FLAGS):
        self.batch_size = FLAGS.batch_size        
        self.latent_dim = FLAGS.latent_dim
        self.num_gpus = FLAGS.num_gpus
        self.num_epochs = FLAGS.num_epochs
        self.start_epochs = int(FLAGS.checkpoint_number) if FLAGS.checkpoint_number is not None else 0
        self.save_freq = FLAGS.save_freq
        self.save_path = FLAGS.save_path
        self.result_output_path = FLAGS.result_output_path if FLAGS.result_output_path is not None else self.save_path
        self.fb_split = FLAGS.fb_split
        self.L = list(map(int, FLAGS.PL.split(',')))
        self.W = list(map(int, FLAGS.PW.split(',')))
        self.level = len(self.L)
        self.dataset = FLAGS.dataset
        self.load_prestored_data = FLAGS.load_prestored_data

        self.range_view = True if FLAGS.partition_mode == 'range' else False
        self.learning_rate = FLAGS.learning_rate
        self.loss_type = FLAGS.loss_type
        self.origin_num_points = list(map(int, FLAGS.cell_max_points.split(',')))
        self.cell_min_points = list(map(int, FLAGS.cell_min_points.split(',')))
        self.DECAY_RATE = 0.8
        self.DECAY_STEP = FLAGS.decay_step
        self.BN_INIT_DECAY = 0.5
        self.BN_DECAY_DECAY_RATE = 0.5
        self.BN_DECAY_DECAY_STEP = float(self.DECAY_STEP)
        self.BN_DECAY_CLIP = 0.99
        self.enable_2ndfold = True
        self.map_size = [8, 8, 8]
        self.encoder_mode = FLAGS.encoder
        self.decoder_mode = FLAGS.decoder
        self.current_level = 0
        self.rotation = FLAGS.rotation
        self.pooling = FLAGS.pooling
        self.top_k = FLAGS.top_k
        self.device_batch_size = int(self.batch_size / self.num_gpus)

        self.num_lasers = 64
        self.group_num_lasers = self.L # can tune
        self.image_height = [int(self.num_lasers / gnl) for gnl in self.group_num_lasers] # image height is PL
        self.image_width = self.W # original 64

        assert(self.batch_size % self.num_gpus == 0)
        self.num_sample_points = 1500
        if FLAGS.mode != "encoder" and FLAGS.mode != "decoder":
            self.nn_config = "autoencoder"
        else:
            self.nn_config = FLAGS.mode

        self.g = tf.Graph()
        self._build_cell(self.g)
        self.eval_loss(self.g)

    def _encoder(self, net, mask, bn_decay, name_scope):

        if self.rotation == True:
            with tf.variable_scope('transform_net1', reuse=tf.AUTO_REUSE) as sc:
                transform = input_transform_net(net, self.is_training, bn_decay, K=3)
            point_cloud_transformed = tf.matmul(net, transform)
            net = point_cloud_transformed
        else:
            transform = None

        with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
            if self.encoder_mode == 'pointnet':
                net = tf_util.point_conv(input_image=net, 
                                         is_training=self.is_training, 
                                         n_filters=[64, 64, 128, self.latent_dim],
                                         bn_decay=bn_decay,
                                         activation_function=tf.nn.relu,
                                         name="encoder")
            elif self.encoder_mode == 'dgcnn':
                net = tf_util.dgcnn(point_cloud=net,
                                    is_training=self.is_training, 
                                    n_filters=[64, 64, 64, 128, self.latent_dim],
                                    k=self.top_k,
                                    activation_function=tf.nn.relu,
                                    bn_decay=bn_decay)
            elif self.encoder_mode == 'magic':
                net = tf_util.magic(net,
                                    is_training=self.is_training,
                                    n_filters=[64, 64, 64, 128, self.latent_dim],
                                    k=self.top_k,
                                    bn_decay=bn_decay,
                                    iterative=False,
                                    step=3)
            elif self.encoder_mode == 'inception':
                net = tf_util.inception_dgcnn(point_cloud=net,
                                              is_training=self.is_training, 
                                              n_filters=[64, 64, 64, 128, self.latent_dim],
                                              max_k=self.top_k,
                                              activation_function=tf.nn.relu,
                                              bn_decay=bn_decay)

        return mask * net, transform

    def _decoder(self, code, mask, bn_decay, name_scope):
        with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
            if self.decoder_mode == 'pointnet':
                x_reconstr = tf_util.fully_connected_decoder(code, 
                                                             self.origin_num_points[self.current_level], 
                                                             self.is_training,
                                                             bn_decay=bn_decay)
            elif self.decoder_mode == 'pointgrid':
                code = tf.expand_dims(code, -2)
                global_code_tile = tf.tile(code, [1, self.origin_num_points[self.current_level], 1])
                local_code_tile = tf_util.fold3d(self.origin_num_points[self.current_level], 
                                                 self.map_size,
                                                 tf.shape(code)[0])
                all_code = tf.concat([local_code_tile, global_code_tile], axis=-1)

                net = tf_util.point_conv(all_code, 
                                         self.is_training, 
                                         n_filters=[256, 128], 
                                         bn_decay=bn_decay,
                                         activation_function=tf.nn.relu,
                                         name="decoder1")
                net = tf_util.point_conv(net, 
                                         self.is_training, 
                                         n_filters=[3],
                                         bn_decay=bn_decay,
                                         activation_function=tf.nn.tanh,
                                         name="decoder2")
                # second folding
                if self.enable_2ndfold:
                    net = tf.concat([local_code_tile, net], axis=-1)
                    net = tf_util.point_conv(net, 
                                             self.is_training, 
                                             n_filters=[258, 256],
                                             # n_filters=[258, 128, 64],
                                             bn_decay=bn_decay, 
                                             activation_function=tf.nn.relu,
                                             name="decoder3")
                    net = tf_util.point_conv(net, 
                                             self.is_training, 
                                             n_filters=[3],
                                             bn_decay=bn_decay,
                                             activation_function=None,
                                             name="decoder4")
                net = tf.reshape(net, [-1, self.origin_num_points[self.current_level], 3])
                net_xy = tf.nn.tanh(net[:,:,0:2])
                net_z = tf.expand_dims(net[:,:,2], -1)
                x_reconstr = tf.concat([net_xy, net_z], -1)
        return mask * x_reconstr

    def _ae(self, pc, mask, bn_decay):

        # ------- encoder -------
        net, transform = self._encoder(pc, mask, bn_decay, name_scope='encoder')
        
        # max pool
        if self.pooling == 'mean':
            code = tf.reduce_mean(net, axis=-2, keepdims=False)
        elif self.pooling == 'max':
            code = tf.reduce_max(net, axis=-2, keepdims=False)
        print (code)

        # ------- decoder ------- 
        x_reconstr = self._decoder(code, mask, bn_decay, name_scope='decoder')
        if self.rotation == True:
            # rotate_matrix = tf.stop_gradient(tf.matrix_inverse(transform))
            rotate_matrix = tf.matrix_inverse(transform)

            return tf.matmul(x_reconstr, rotate_matrix), code
        else:
            return x_reconstr, code

    def _pixel(self, embedding, bn_decay):
        
        net = conv2d(net, n_filter, [1,1],
             padding='VALID', stride=[1,1],
             activation_fn=tf.nn.relu,
             bn=True, is_training=is_training,
             bn_decay=bn_decay,
             scope=name+'_'+str(idx))


    def _build_cell(self, g):
        with g.as_default():
            # MLP: all_points [batch, 200, 2] -> MLP -> node_feature [batch, 200, 10]
            self.pc = tf.placeholder(tf.float32, [None, self.origin_num_points[self.current_level], 3])
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.meta = tf.placeholder(tf.int32, [None])
            mask = tf.sequence_mask(self.meta, maxlen=self.origin_num_points[self.current_level], dtype=tf.float32)
            self.mask = tf.expand_dims(mask, -1)
        
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)

            bn_decay = self.get_bn_decay(batch)

            def calculate_chamfer(recon, orig, num_points):
                x_reconstr, pc = tf.expand_dims(recon[:num_points], 0), tf.expand_dims(orig[:num_points], 0)
                cost_p1_p2, _, cost_p2_p1, _ = nn_distance(x_reconstr, pc)
                chamfer_return = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
                chamfer_return = tf.where(tf.is_nan(chamfer_return), 0., chamfer_return)
                return chamfer_return

            # reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            learning_rate = self.get_learning_rate(batch)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

            tower_grads = []
            pred_gpu = []
            latent_gpu = []
            total_emd_loss, total_chamfer_loss = [], []
            for i in range(self.num_gpus):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    with tf.device('/gpu:%d'%(i)), tf.name_scope('gpu_%d'%(i)) as scope:
                        pc_batch = tf.slice(self.pc, [i*self.device_batch_size,0,0], [self.device_batch_size,-1,-1])
                        meta_batch = tf.slice(self.meta, [i*self.device_batch_size], [self.device_batch_size])
                        mask_batch = tf.slice(self.mask, [i*self.device_batch_size,0,0], [self.device_batch_size,-1,-1])

                        x_reconstr, code = self._ae(pc_batch, mask_batch, bn_decay)

                        match = approx_match(pc_batch, x_reconstr)
                        emd_loss = tf.reduce_mean(match_cost(pc_batch, x_reconstr, match))
            
                        # self.chamfer_loss = calculate_chamfer(self.x_reconstr, self.pc)
                        chamfer_loss = tf.reduce_mean(tf.map_fn(lambda x: calculate_chamfer(x[0], x[1], x[2]), 
                                                                     (x_reconstr, pc_batch, meta_batch), dtype=tf.float32))
                        #  -------  loss + optimization  ------- 
                        if self.loss_type == "emd":
                            reconstruction_loss = emd_loss
                        elif self.loss_type == "chamfer":
                            reconstruction_loss = chamfer_loss

                        grads = optimizer.compute_gradients(reconstruction_loss)
                        tower_grads.append(grads)
                        pred_gpu.append(x_reconstr)
                        latent_gpu.append(code)
                        total_emd_loss.append(emd_loss)
                        total_chamfer_loss.append(chamfer_loss)
                        
            self.x_reconstr = tf.concat(pred_gpu, 0)
            self.latent_code = tf.concat(latent_gpu, 0)
            self.emd_loss = tf.reduce_mean(total_emd_loss)
            self.chamfer_loss = tf.reduce_mean(total_chamfer_loss)
            grads = average_gradients(tower_grads)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.apply_gradients(grads, global_step=batch)
            
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            self.sess = tf.Session(config=config)

            self.sess.run(tf.global_variables_initializer())
            # self.saver = tf.train.Saver(tf.trainable_variables() + tf.get_collection_ref("batch_norm_non_trainable_variables_co‌​llection"))
            self.saver = tf.train.Saver()

    def eval_loss(self, g):
        '''
        this function evaluate the whole sweep loss (not cell) that downsampled to self.num_sample_points.
        '''
        with g.as_default():
            self.pred_meta = tf.placeholder(tf.int32, [None])
            self.fetched_pred = tf.placeholder(tf.float32, [None, self.num_sample_points, 3])
            self.fetched_orig = tf.placeholder(tf.float32, [None, self.num_sample_points, 3])

            def loss(recon, orig, num_points):
                recon = tf.gather(recon, tf.range(num_points))
                orig, recon = tf.expand_dims(orig, 0), tf.expand_dims(recon, 0)
                match = approx_match(orig, recon)
                emd_loss = tf.reduce_mean(match_cost(orig, recon, match))

                cost_p1_p2, _, cost_p2_p1, _ = nn_distance(recon, orig)
                chamfer_loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
                return emd_loss, chamfer_loss

            emd, chamfer = tf.map_fn(lambda x: loss(x[0], x[1], x[2]), 
                           (self.fetched_pred, self.fetched_orig, self.pred_meta), dtype=(tf.float32, tf.float32))
            self.eval_emd = emd
            self.eval_chamfer = chamfer

    def train(self, point_cell, mode='foreground'):
        
        dataset = collections.namedtuple('dataset', [
                            'train_set', 'train_meta', 'valid_set', 'valid_meta',
                            'test_set', 'test_meta'])
        train_point, valid_point, test_point, \
        train_meta, valid_meta, test_meta = self.preprocess(point_cell, self.current_level, mode)
        data = dataset(train_set=train_point,
                       train_meta=train_meta,
                       valid_set=valid_point,
                       valid_meta=valid_meta,
                       test_set=test_point,
                       test_meta=test_meta)
        print ('-------- train level {} --------'.format(self.current_level))
        self.train_level(data, point_cell, self.current_level, mode)

    def train_level(self, dataset, point_cell, level_idx, mode):
        record_loss = []

        train_point, train_meta = dataset.train_set, dataset.train_meta
        valid_point, valid_meta = dataset.valid_set, dataset.valid_meta
        test_point, test_meta = dataset.test_set, dataset.test_meta
        num_batches = train_point.shape[0] // self.batch_size
        ckpt_path = util.create_dir(osp.join(self.save_path, 'level_%s' % (self.current_level)))
        
        for epoch in range(self.num_epochs):

            train_idx = np.arange(0, len(dataset.train_set))
            np.random.shuffle(train_idx)
            current_point = train_point[train_idx]
            current_meta = train_meta[train_idx]

            record_mean, record_var, record_mse = [], [], []

            start = time.time()
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx+1) * self.batch_size

                X = current_point[start_idx:end_idx]
                meta = current_meta[start_idx:end_idx]

                feed_dict={self.pc: X, self.meta: meta, self.is_training: True}
                op = [self.train_op, 
                      self.emd_loss,
                      self.chamfer_loss,
                      self.x_reconstr,]
                _, emd_loss, chamfer_loss, recon = self.sess.run(op, feed_dict)
                if self.loss_type == 'emd':
                    record_loss.append(emd_loss)
                elif self.loss_type == 'chamfer':
                    record_loss.append(chamfer_loss)
                mean, var, mse = util.dist_stat(recon, X, meta, self.origin_num_points[level_idx])
                record_mean.append(mean)
                record_var.append(var)
                record_mse.append(mse)

                compression_ratio = (self.batch_size*self.latent_dim) / (np.sum(meta)*3.0)
                if batch_idx % 100 == 0:
                    print ("iteration/epoch: {}/{}, optimizing {} loss, emd loss: {}, chamfer loss: {}".format(
                        batch_idx, epoch, self.loss_type, emd_loss, chamfer_loss))
                    print ("mean: %.6f, var: %.6f, mse: %.6f, compression ratio: %.6f\n" % 
                                                                      (np.array(record_mean).mean(), 
                                                                       np.array(record_var).mean(), 
                                                                       np.array(record_mse).mean(),
                                                                       compression_ratio))
            end = time.time()
            print("epoch: {}, elapsed time: {}".format(epoch, end - start))

            if epoch % self.save_freq == 0:
                if self.fb_split == True:
                    ckpt_name = osp.join(ckpt_path, 'model-%s-%s.ckpt' % (mode, str(epoch)))
                else:
                    ckpt_name =  osp.join(ckpt_path, 'model-%s.ckpt' % (str(epoch)))
                
                self.saver.save(self.sess, ckpt_name)

                # evaluation
                self.evaluate(valid_point, valid_meta, ckpt_path, test=False)
                self.evaluate(test_point, test_meta, ckpt_path, test=True)

                def visualize_sample(sample_points, sample_meta, orig_points, orig_meta, filename=None):
                    
                    meta_nums = sample_meta['num_points'].values.astype(int)
                    if (meta_nums >= self.cell_min_points[level_idx]).all():

                        sample_generated, sample_emd, sample_chamfer = [], [], []
                        for idx in range(0, sample_points.shape[0], self.batch_size):
                            start_idx, end_idx = idx, min(idx+self.batch_size, sample_points.shape[0])
                            if end_idx-start_idx == self.batch_size:
                                fetched_pc = sample_points[start_idx:end_idx]
                                fetched_meta = meta_nums[start_idx:end_idx]
                            else:
                                pad_size = self.batch_size - (end_idx - start_idx)
                                pad_shape = [pad_size] + list(sample_points[start_idx:end_idx].shape[1:])
                                fetched_pc = np.concatenate([sample_points[start_idx:end_idx], np.zeros(pad_shape)], axis=0)
                                fetched_meta = np.concatenate([meta_nums[start_idx:end_idx], np.zeros(pad_size)], axis=0)

                            feed_dict = {self.pc: fetched_pc, 
                                         self.meta: fetched_meta, 
                                         self.is_training: False}
                            sg, se, sc = self.sess.run([self.x_reconstr, 
                                                        self.emd_loss, 
                                                        self.chamfer_loss], feed_dict)
                            sample_generated.append(sg)
                            sample_emd.append(se)
                            sample_chamfer.append(sc)
                        sample_emd, sample_chamfer = np.mean(sample_emd), np.mean(sample_chamfer)
                        sample_generated = np.concatenate(sample_generated, 0)

                    else:
                        print ("compressed points should above min points settings!")
                        raise

                    reconstruction_sample = point_cell.reconstruct_scene(sample_generated, sample_meta)
                    reconstruction_orig = point_cell.reconstruct_scene(orig_points, orig_meta)
                    reconstruction = np.concatenate([reconstruction_sample, reconstruction_orig], 0)

                    if filename == None:
                        sweep = point_cell.test_sweep[point_cell.sample_idx]
                    elif filename == 'foreground':
                        sweep = point_cell.test_f_sweep
                    elif filename == 'background':
                        sweep = point_cell.test_b_sweep

                    mean, var, mse = util.sweep_stat(reconstruction, sweep)
                    print ('sampled sweep emd loss: {}, chamfer loss: {}, MSE: {}'.format(sample_emd,
                                                                                          sample_chamfer,
                                                                                          mse))
                    print ('save reconstructed sweep')
                    full_filename = 'orig_'+str(epoch) if filename == None else 'orig_{}_{}'.format(filename, str(epoch))
                    util.visualize_3d_points(sweep, dir=ckpt_path, filename=full_filename)
                    full_filename = 'recon_'+str(epoch) if filename == None else 'recon_{}_{}'.format(filename, str(epoch))
                    util.visualize_3d_points(reconstruction, dir=ckpt_path, filename=full_filename)
                    return reconstruction, sweep
                
                # plot
                if self.fb_split == True:
                    sample_points, sample_meta, _, orig_points, orig_meta, _ = self.extract_point(point_cell.sample_points_f, level_idx)
                    rf, sf = visualize_sample(sample_points, sample_meta, orig_points, orig_meta, 'foreground')

                    sample_points, orig_points, _, orig_points, orig_meta, _ = self.extract_point(point_cell.sample_points_b, level_idx)
                    rb, sb = visualize_sample(sample_points, sample_meta, orig_points, orig_meta, 'background')
                    reconstruction = np.concatenate([rf, rb], 0)
                    sweep = np.concatenate([sf, sb], 0)
                    util.visualize_3d_points(sweep, dir=ckpt_path, filename='orig_{}'.format(str(epoch)))
                    util.visualize_3d_points(reconstruction, dir=ckpt_path, filename='recon_{}'.format(str(epoch)))
                else:
                    sample_points, sample_meta, _, orig_points, orig_meta, _ = self.extract_point(point_cell.sample_points, level_idx)
                    visualize_sample(sample_points, sample_meta, orig_points, orig_meta)
                

        # check testing accuracy
        self.evaluate(test_point, test_meta, ckpt_path, test=True)

    def train_image(self, point_cell):
        '''
        training on pixel level, under construction
        '''
        train_point, valid_point, test_point, \
        train_meta, valid_meta, test_meta = self.preprocess_image(point_cell, self.current_level, mode)

    def get_learning_rate(self, batch):
        learning_rate = tf.train.exponential_decay(
                            self.learning_rate,  # Base learning rate.
                            batch * self.batch_size,  # Current index into the dataset.
                            self.DECAY_STEP,          # Decay step.
                            self.DECAY_RATE,          # Decay rate.
                            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
        return learning_rate

    def get_bn_decay(self, batch):
        bn_momentum = tf.train.exponential_decay(
                          self.BN_INIT_DECAY,
                          batch*self.batch_size,
                          self.BN_DECAY_DECAY_STEP,
                          self.BN_DECAY_DECAY_RATE,
                          staircase=True)
        bn_decay = tf.minimum(self.BN_DECAY_CLIP, 1 - bn_momentum)
        return bn_decay


    def evaluate(self, valid_point, valid_meta, save_path, test=False):
        '''
        this function evaluate cell-level emd loss and chamfer loss, and sweep-level mean, var, mse,
        note that this only evaluate one sample sweep
        '''
        valid_emd_loss, valid_chamfer_loss = [], []
        mean_arr, var_arr, mse_arr = [], [], []
        save_emd, save_chamfer = [], []
        save_mean, save_var, save_mse = [], [], []
        for i in range(len(valid_point)//self.batch_size):
            valid_s, valid_e = i*self.batch_size, (i+1)*self.batch_size
            feed_dict = {self.pc: valid_point[valid_s:valid_e],
                         self.meta: valid_meta[valid_s:valid_e],
                         self.is_training: False}

            vel, vcl, recon = self.sess.run([self.emd_loss, self.chamfer_loss, self.x_reconstr], feed_dict)
            mean, var, mse = util.dist_stat(recon, 
                                            valid_point[valid_s:valid_e], 
                                            valid_meta[valid_s:valid_e], self.origin_num_points[self.current_level])
            mean_arr.append(mean)
            var_arr.append(var)
            mse_arr.append(mse)
            valid_emd_loss.append(vel)
            valid_chamfer_loss.append(vcl)

        save_emd.append(np.array(valid_emd_loss).mean())
        save_chamfer.append(np.array(valid_chamfer_loss).mean())
        save_mean.append(np.array(mean_arr).mean())
        save_var.append(np.array(var_arr).mean())
        save_mse.append(np.array(mse_arr).mean())

        if test == True:
            print ('Testing emd loss: {}, Testing chamfer loss: {}'.format(np.array(valid_emd_loss).mean(),
                                                                           np.array(valid_chamfer_loss).mean()))
            print ('mean: %.6f, var: %.6f, mse: %.6f' % (np.array(mean_arr).mean(),
                                                             np.array(var_arr).mean(),
                                                             np.array(mse_arr).mean()))

            with open(os.path.join(save_path, 'mean.txt'), 'a') as myfile:
                np.savetxt(myfile, save_mean)
            with open(os.path.join(save_path, 'var.txt'), 'a') as myfile:
                np.savetxt(myfile, save_var)
            with open(os.path.join(save_path, 'mse.txt'), 'a') as myfile:
                np.savetxt(myfile, save_mse)
            with open(os.path.join(save_path, 'emd.txt'), 'a') as myfile:
                np.savetxt(myfile, save_emd)
            with open(os.path.join(save_path, 'chamfer.txt'), 'a') as myfile:
                np.savetxt(myfile, save_chamfer)
            
        else:
            print ('validation emd loss: {}, validation chamfer loss: {}'.format(np.array(valid_emd_loss).mean(),
                                                                                 np.array(valid_chamfer_loss).mean()))
            print ('mean: %.6f, var: %.6f, mse: %.6f' % (np.array(mean_arr).mean(), 
                                                         np.array(var_arr).mean(), 
                                                         np.array(mse_arr).mean()))

    def sub_sample(self, pred, orig, repeat=1):
        '''
        this function is called by evaluate_sweep, sub_sample evaluate the sweep-level downsampled loss, the graph
        is constructed via function eval_loss
        '''
        vel, vcl = [], []
        for i in range(repeat):
            if pred.shape[0] > self.num_sample_points:
                pred_sample_idx = np.random.choice(pred.shape[0], self.num_sample_points, replace=False)
                fetched_pred = np.expand_dims(pred[pred_sample_idx], 0)
            else:
                pred_sample_idx = np.arange(pred.shape[0])
                pad_size = self.num_sample_points - pred.shape[0]
                padding = np.zeros([pad_size, 3])
                fetched_pred = np.expand_dims(np.concatenate([pred[pred_sample_idx], padding], 0), 0)
            orig_sample_idx = np.random.choice(orig.shape[0], self.num_sample_points, replace=False)
            
            feed_dict = {self.fetched_pred: fetched_pred,
                         self.pred_meta: np.array([min(pred.shape[0], self.num_sample_points)]),
                         self.fetched_orig: np.expand_dims(orig[orig_sample_idx], 0)}
            sample_vel, sample_vcl = self.sess.run([self.eval_emd, self.eval_chamfer], feed_dict)
            vel.append(sample_vel)
            vcl.append(sample_vcl)
        return np.mean(vel), np.mean(vcl)
        

    def evaluate_sweep(self, point_cell, compress='random', num_points=6):
        '''
        this is the main function for evaluation, which is the function to obtain the evaluated results 
        filled in paper table, the function contains kmeans, random, octree, learning based method (autoencoder).
        all displayed results are sweep-level on testing set.
        '''
        if self.level > 1:
            level_idx = self.label_level
            if self.combination == 'down':
                points = self.group_multi_down(point_cell.test_cell, level_idx)
            elif self.combination == 'up':
                points = self.group_multi_up(point_cell.test_cell, level_idx)
            elif self.combination == 'sandwich':
                points = self.group_multi_middle(point_cell.test_cell)
                level_idx = -2
            num_cell = len(points)
        else:
            points = point_cell.test_cell[0]
            num_cell = len(points)
            level_idx = 0

        if self.range_view == False:
            sweep_size = self.L[level_idx] * self.W[level_idx]
        else:
            sweep_size = point_cell.image_height[level_idx] * point_cell.image_width[level_idx]
        save_emd_all, save_chamfer_all = [], []
        save_mean_all, save_var_all, save_mse_all = [], [], []
        save_emd_part, save_chamfer_part = [], []
        save_mean_part, save_var_part, save_mse_part = [], [], []
        test_batch_size = self.batch_size

        for i in tqdm(range(0, num_cell, sweep_size)):
            s, e = i, i+sweep_size
            sweep_compress, compress_meta, compress_num, sweep_orig, orig_meta, orig_num = self.extract_sweep(points, s, e, level_idx)
            if self.level > 1:
                gt_points = point_cell.reconstruct_scene(sweep_compress[:,level_idx,...], compress_meta) # debug here
                orig_points = point_cell.reconstruct_scene(sweep_orig[:,level_idx,...], orig_meta)
            else:
                gt_points = point_cell.reconstruct_scene(sweep_compress, compress_meta)
                if sweep_orig.shape[0] > 0:
                    orig_points = point_cell.reconstruct_scene(sweep_orig, orig_meta)
            import numpy as np
            orig_all = np.concatenate([gt_points, orig_points], axis=0) if sweep_orig.shape[0] > 0 else gt_points

            num_compress = compress_meta.as_matrix(columns=['num_points']).squeeze().astype(int)
            total_compress_num = orig_num.sum() + num_compress.shape[0]*num_points
            total_points = []
            total_vel, total_vcl = 0, 0
            if compress == 'random':
                idx = np.random.choice(orig_all.shape[0], total_compress_num, replace=False)
                pred_all = orig_all[idx]
            elif compress == 'octree':
                Points_set = tuple(map(tuple, orig_all))
                octree_dict = {}
                world_center_x = (np.max(orig_all[:,0])+np.min(orig_all[:,0]))//2
                world_center_y = (np.max(orig_all[:,1])+np.min(orig_all[:,1]))//2
                world_center_z = 0.0
                world_center = (world_center_x, world_center_y, world_center_z)
                world_size = max(np.max(orig_all), abs(np.min(orig_all)))*2
                octree.octree_func(octree_dict, Points_set, orig_all.shape[0], 9, world_size, world_center)

                p = []
                for key, value in octree_dict.items():
                    point_subspace = orig_all[np.asarray(value, dtype=np.int32),:]
                    p_center = np.mean(point_subspace, axis=0)
                    p.append(p_center)
                idx = np.random.choice(len(p), total_compress_num, replace=False)
                pred_all = np.array(p)[idx]
            else:
                for j in range(len(sweep_compress)//test_batch_size + 1):
                    valid_s, valid_e = j*test_batch_size, min((j+1)*test_batch_size, len(sweep_compress))
                    # if compress == 'random':
                    #     recon = []
                    #     for k in range(valid_e-valid_s):
                    #         meta = compress_num[valid_s:valid_e][k]
                    #         idx = np.random.choice(meta, num_points, replace=False)
                    #         recon.append(sweep_compress[valid_s:valid_e][k][idx])
                    #     recon = np.array(recon)
                    if compress == 'kmeans':
                        recon = []
                        for k in range(valid_e-valid_s):
                            centroids = KMeans(n_clusters=num_points, random_state=0).fit(sweep_compress[valid_s:valid_e][k])
                            recon.append(centroids.cluster_centers_)
                        recon = np.array(recon)
                    elif compress == 'autoencoder':
                        if self.level > 1:
                            if valid_e-valid_s == test_batch_size:
                                fetched_pc = sweep_compress[valid_s:valid_e]
                                fetched_pmeta = compress_num[valid_s:valid_e]
                            else:
                                pad_size = test_batch_size - (valid_e - valid_s)
                                pad_shape = [pad_size] + list(sweep_compress[valid_s:valid_e].shape[1:])
                                fetched_pc = np.concatenate([sweep_compress[valid_s:valid_e], np.zeros(pad_shape)], axis=0)
                                pmeta_shape = [pad_size] + ([self.level] if self.combination == 'down' else list(compress_num.shape[1:]))
                                fetched_pmeta = np.concatenate([compress_num[valid_s:valid_e], np.zeros(pmeta_shape)], axis=0)
                            feed_dict = self.feed_data(fetched_pc, fetched_pmeta, is_training=False)
                        else:
                            if valid_e - valid_s == test_batch_size:
                                fetched_pc = sweep_compress[valid_s:valid_e]
                                fetched_meta = compress_num[valid_s:valid_e]
                            else:
                                pad_size = test_batch_size - (valid_e - valid_s)
                                pad_shape = [pad_size] + list(sweep_compress[valid_s:valid_e].shape[1:])
                                fetched_pc = np.concatenate([sweep_compress[valid_s:valid_e], np.zeros(pad_shape)], axis=0)
                                fetched_meta = np.concatenate([compress_num[valid_s:valid_e], np.zeros(pad_size)], axis=0)
                            feed_dict = {self.pc: fetched_pc,
                                         self.meta: fetched_meta,
                                         self.is_training: False}
                        vel, vcl, recon = self.sess.run([self.emd_loss, self.chamfer_loss, self.x_reconstr], feed_dict)
                        # emd loss and chamfer loss return mean of batch, should multiply batch_size
                        total_vel += vel*test_batch_size
                        total_vcl += vcl*test_batch_size
                    total_points.append(recon)

                # orig = np.concatenate([sweep_compress[~np.all(sweep_compress == 0, axis=2)], sweep_orig[~np.all(sweep_orig == 0, axis=2)]], axis=0)
                total_points = np.concatenate(total_points, axis=0)
                pred_points = point_cell.reconstruct_scene(total_points, compress_meta)
            
                pred_all = np.concatenate([pred_points, orig_points], axis=0) if orig_points.shape[0] > 0 else pred_points
            
            mean_all, var_all, mse_all = util.sweep_stat(pred_all, orig_all)
            sample_vel_all, sample_vcl_all = self.sub_sample(pred_all, orig_all, repeat=10)
            
            if compress != 'random' and compress != 'octree':
                pred_part, orig_part = pred_points, gt_points
                mean_part, var_part, mse_part = util.sweep_stat(pred_part, orig_part)
                sample_vel_part, sample_vcl_part = self.sub_sample(pred_part, orig_part, repeat=10)

                save_emd_part.append(sample_vel_part)
                save_chamfer_part.append(sample_vcl_part)
                save_mean_part.append(mean_part)
                save_var_part.append(var_part)
                save_mse_part.append(mse_part)

            #save_emd.append(total_vel / (orig_meta.sum()+compress_meta.sum()))
            #save_chamfer.append(total_vcl / (orig_meta.sum()+compress_meta.sum()))
            save_emd_all.append(sample_vel_all)
            save_chamfer_all.append(sample_vcl_all)
            save_mean_all.append(mean_all)
            save_var_all.append(var_all)
            save_mse_all.append(mse_all)

        print ("all points evaluation statistics:")
        print ('testing sweep mean emd loss: {}, mean chamfer loss: {}'.format(np.array(save_emd_all).mean(),
                                                                               np.array(save_chamfer_all).mean()))
        print ('mean: %.6f, var: %.6f, mse: %.6f\n' % (np.array(save_mean_all).mean(), 
                                                     np.array(save_var_all).mean(), 
                                                     np.array(save_mse_all).mean()))

        if compress != 'random' and compress != 'octree':
            print ("compressed points evaluation statistics:")
            print ('testing sweep mean emd loss: {}, mean chamfer loss: {}'.format(np.array(save_emd_part).mean(),
                                                                                   np.array(save_chamfer_part).mean()))
            print ('mean: %.6f, var: %.6f, mse: %.6f\n' % (np.array(save_mean_part).mean(), 
                                                         np.array(save_var_part).mean(), 
                                                         np.array(save_mse_part).mean()))

        cell_rate, sweep_rate = point_cell.test_compression_rate(level_idx=level_idx)

        emd_all = ['emd_all', np.array(save_emd_all).mean()]
        chamfer_all = ['chamfer_all', np.array(save_chamfer_all).mean()]
        mean_all = ['mean_all', np.array(save_mean_all).mean()]
        var_all = ['var_all', np.array(save_var_all).mean()]
        mse_all = ['mse_all', np.array(save_mse_all).mean()]
        emd_part = ['emd_part', np.array(save_emd_part).mean()]
        chamfer_part = ['chamfer_part', np.array(save_chamfer_part).mean()]
        mean_part = ['mean_part', np.array(save_mean_part).mean()]
        var_part = ['var_part', np.array(save_var_part).mean()]
        mse_part = ['mse_part', np.array(save_mse_part).mean()]
        cell_rate = ['cell_compression_rate', cell_rate]
        sweep_rate = ['sweep_compression_rate', sweep_rate]

        print("self.result_output_path: {}".format(self.result_output_path))
        np.savetxt(self.result_output_path + '/result.csv', (emd_all, chamfer_all, 
                    mean_all, var_all, mse_all, emd_part, chamfer_part, mean_part, 
                    var_part, mse_part, cell_rate, sweep_rate), delimiter=',', fmt='%s')

    def plot_hdmap(self, point_cell, center, ckpt_name, mode, num_points=6, level_idx=0):
        '''
        this is the plot function that plot hdmap in testing set, This plot function 
        is not stable, [TODO]: need to include octree
        '''
        points = point_cell.test_cell[0]
        test_batch_size = self.batch_size
        if mode == 'autoencoder':
            ckpt_path = util.create_dir(osp.join(self.save_path, 'level_%s' % (self.current_level)))
            self.saver.restore(self.sess, os.path.join(ckpt_path, ckpt_name))

        if self.range_view == False:
            sweep_size = self.L[level_idx] * self.W[level_idx]
        else:
            sweep_size = point_cell.image_height[level_idx] * point_cell.image_width[level_idx]

        hd_map = []
        hd_map_pred = []
        for i in tqdm(range(0, len(points), sweep_size)):
            s, e = i, i+sweep_size
            sweep_compress, compress_meta, sweep_orig, orig_meta = self.extract_sweep(points, s, e, level_idx)
            gt_points = point_cell.reconstruct_scene(sweep_compress, compress_meta)
            orig_points = point_cell.reconstruct_scene(sweep_orig, orig_meta)
            orig_all = np.concatenate([gt_points, orig_points], axis=0)
            center = np.mean(orig_all, 0)
            sweep_compress -= center
            sweep_orig -= center

            compress_nums = compress_meta.as_matrix(columns=['num_points']).squeeze().astype(int)
            orig_nums = orig_meta.as_matrix(columns=['num_points']).squeeze().astype(int)
            total_points = []
            total_vel, total_vcl = 0, 0

            for j in range(len(sweep_compress)//test_batch_size + 1):
                valid_s, valid_e = j*test_batch_size, min((j+1)*test_batch_size, len(sweep_compress))
                if mode == 'random':
                    recon = []
                    for k in range(valid_e-valid_s):
                        meta = compress_nums[valid_s:valid_e][k]
                        idx = np.random.choice(meta, num_points, replace=False)
                        recon.append(sweep_compress[valid_s:valid_e][k][idx])
                    recon = np.array(recon)
                elif mode == 'kmeans':
                    recon = []
                    for k in range(valid_e-valid_s):
                        centroids = KMeans(n_clusters=num_points, random_state=0).fit(sweep_compress[valid_s:valid_e][k])
                        recon.append(centroids.cluster_centers_)
                    recon = np.array(recon)
                elif mode == 'autoencoder':
                    if valid_e - valid_s == test_batch_size:
                        fetched_pc = sweep_compress[valid_s:valid_e]
                        fetched_meta = compress_nums[valid_s:valid_e]
                    else:
                        pad_size = test_batch_size - (valid_e - valid_s)
                        pad_shape = [pad_size] + list(sweep_compress[valid_s:valid_e].shape[1:])
                        fetched_pc = np.concatenate([sweep_compress[valid_s:valid_e], np.zeros(pad_shape)], axis=0)
                        fetched_meta = np.concatenate([compress_nums[valid_s:valid_e], np.zeros(pad_size)], axis=0)
                    feed_dict = {self.pc: fetched_pc,
                                 self.meta: fetched_meta,
                                 self.is_training: False}
                    vel, vcl, recon = self.sess.run([self.emd_loss, self.chamfer_loss, self.x_reconstr], feed_dict)
                    # emd loss and chamfer loss return mean of batch, should multiply batch_size
                total_points.append(recon)

            total_points = np.concatenate(total_points, axis=0)
            pred_points = point_cell.reconstruct_scene(total_points, compress_meta) + center
            
            pred_all = np.concatenate([pred_points, orig_points+center], axis=0)
            hd_map.append(orig_all)
            hd_map_pred.append(pred_all)

        hd_map = np.concatenate(hd_map, axis=0)
        hd_map_pred = np.concatenate(hd_map_pred, axis=0)
        
        if mode == 'autoencoder':
            filename = self.encoder_mode
        else:
            filename = mode
        util.visualize_3d_points(hd_map, filename=filename)


    def plot_sweep(self, point_cell, idx, ckpt_name, mode, num_points=6, level_idx=0):
        '''
        this is sample plot function that plot kitti in testing set, spcified by idx
        '''
        points = point_cell.test_cell[0]
        test_batch_size = self.batch_size
        if mode == 'autoencoder':
            ckpt_path = util.create_dir(osp.join(self.save_path, 'level_%s' % (self.current_level)))
            self.saver.restore(self.sess, os.path.join(ckpt_path, ckpt_name))

        if self.range_view == False:
            sweep_size = self.L[level_idx] * self.W[level_idx]
        else:
            sweep_size = point_cell.image_height[level_idx] * point_cell.image_width[level_idx]
        s, e = idx*sweep_size, (idx+1)*sweep_size
        sweep_compress, compress_meta, sweep_orig, orig_meta = self.extract_sweep(points, s, e, level_idx)
        gt_points = point_cell.reconstruct_scene(sweep_compress, compress_meta)
        orig_points = point_cell.reconstruct_scene(sweep_orig, orig_meta)
        orig_all = np.concatenate([gt_points, orig_points], axis=0)

        compress_nums = compress_meta.as_matrix(columns=['num_points']).squeeze().astype(int)
        orig_nums = orig_meta.as_matrix(columns=['num_points']).squeeze().astype(int)
        total_compress_num = orig_nums.sum() + compress_nums.shape[0]*num_points
        total_points = []
        total_vel, total_vcl = 0, 0
        if mode == 'random':
            idx = np.random.choice(orig_all.shape[0], total_compress_num, replace=False)
            pred_all = orig_all[idx]
        elif mode == 'octree':
            Points_set = tuple(map(tuple, orig_all))
            octree_dict = {}
            world_center_x = (np.max(orig_all[:,0])+np.min(orig_all[:,0]))//2
            world_center_y = (np.max(orig_all[:,1])+np.min(orig_all[:,1]))//2
            world_center_z = 0.0
            world_center = (world_center_x, world_center_y, world_center_z)
            world_size = max(np.max(orig_all), abs(np.min(orig_all)))*2
            octree.octree_func(octree_dict, Points_set, orig_all.shape[0], 9, world_size, world_center)

            p = []
            for key, value in octree_dict.items():
                point_subspace = orig_all[np.asarray(value, dtype=np.int32),:]
                p_center = np.mean(point_subspace, axis=0)
                p.append(p_center)
            idx = np.random.choice(len(p), total_compress_num, replace=False)
            pred_all = np.array(p)[idx]
        else:
            for j in range(len(sweep_compress)//test_batch_size + 1):
                valid_s, valid_e = j*test_batch_size, min((j+1)*test_batch_size, len(sweep_compress))
                # if mode == 'random':
                #     recon = []
                #     for k in range(valid_e-valid_s):
                #         meta = compress_nums[valid_s:valid_e][k]
                #         idx = np.random.choice(meta, num_points, replace=False)
                #         recon.append(sweep_compress[valid_s:valid_e][k][idx])
                #     recon = np.array(recon)
                if mode == 'kmeans':
                    recon = []
                    for k in range(valid_e-valid_s):
                        centroids = KMeans(n_clusters=num_points, random_state=0).fit(sweep_compress[valid_s:valid_e][k])
                        recon.append(centroids.cluster_centers_)
                    recon = np.array(recon)
                elif mode == 'autoencoder':
                    if valid_e - valid_s == test_batch_size:
                        fetched_pc = sweep_compress[valid_s:valid_e]
                        fetched_meta = compress_nums[valid_s:valid_e]
                    else:
                        pad_size = test_batch_size - (valid_e - valid_s)
                        pad_shape = [pad_size] + list(sweep_compress[valid_s:valid_e].shape[1:])
                        fetched_pc = np.concatenate([sweep_compress[valid_s:valid_e], np.zeros(pad_shape)], axis=0)
                        fetched_meta = np.concatenate([compress_nums[valid_s:valid_e], np.zeros(pad_size)], axis=0)
                    feed_dict = {self.pc: fetched_pc,
                                 self.meta: fetched_meta,
                                 self.is_training: False}
                    vel, vcl, recon = self.sess.run([self.emd_loss, self.chamfer_loss, self.x_reconstr], feed_dict)
                    # emd loss and chamfer loss return mean of batch, should multiply batch_size
                total_points.append(recon)

            total_points = np.concatenate(total_points, axis=0)
            pred_points = point_cell.reconstruct_scene(total_points, compress_meta)
            pred_all = np.concatenate([pred_points, orig_points], axis=0)
        
        if mode == 'autoencoder':
            filename = self.encoder_mode
        else:
            filename = mode
        util.visualize_3d_points(pred_all, filename=filename)


    def extract_sweep(self, points, start_idx, end_idx, level_idx=0):
        level_points = points.iloc[start_idx:end_idx]
        sample_points_df = level_points[level_points['num_points'] >= self.cell_min_points[level_idx]]
        sample_points = sample_points_df.as_matrix(columns=['points'])
        sample_points = np.array(list(sample_points.squeeze(1)))
        sample_num = sample_points_df.as_matrix(columns=['num_points']).squeeze().astype(int)
        sample_meta = sample_points_df

        orig_points_df = level_points[(level_points['num_points']<self.cell_min_points[level_idx]) & (level_points['num_points']>0)]
        orig_points = orig_points_df.as_matrix(columns=['points'])
        orig_points = np.array(list(orig_points.squeeze(1)))
        orig_num = orig_points_df.as_matrix(columns=['num_points']).squeeze().astype(int)
        orig_meta = orig_points_df

        return sample_points, sample_meta, sample_num, orig_points, orig_meta, orig_num


    def extract_point(self, points, level_idx=0):
        level_points = points[level_idx]
        sample_points_df = level_points[level_points['num_points'] >= self.cell_min_points[level_idx]]
        sample_points = sample_points_df.as_matrix(columns=['points'])
        sample_points = np.array(list(sample_points.squeeze(1)))
        sample_num = sample_points_df.as_matrix(columns=['num_points']).squeeze().astype(int)
        sample_meta = sample_points_df

        orig_points_df = level_points[(level_points['num_points']<self.cell_min_points[level_idx]) & (level_points['num_points']>0)]
        orig_points = orig_points_df.as_matrix(columns=['points'])
        orig_points = np.array(list(orig_points.squeeze(1)))
        orig_num = orig_points_df.as_matrix(columns=['num_points']).squeeze().astype(int)
        orig_meta = orig_points_df

        return sample_points, sample_meta, sample_num, orig_points, orig_meta, orig_num

    def preprocess(self, point_cell, level_idx, mode='foreground'):
        if self.fb_split == True:
            if mode == 'foreground':
                train_point, train_meta, train_num, _, _, _ = self.extract_point(point_cell.train_cell_f, level_idx)
                valid_point, valid_meta, valid_num, _, _, _ = self.extract_point(point_cell.valid_cell_f, level_idx)
                test_point, test_meta, test_num, _, _, _ = self.extract_point(point_cell.test_cell_f, level_idx)

            elif mode == 'background':
                train_point, train_meta, train_num, _, _, _ = self.extract_point(point_cell.train_cell_b, level_idx)
                valid_point, valid_meta, valid_num, _, _, _ = self.extract_point(point_cell.valid_cell_b, level_idx)
                test_point, test_meta, test_num, _, _, _ = self.extract_point(point_cell.test_cell_b, level_idx)

            else:
                print ("unkonwn mode {}, should be either foreground or background!" % mode)
                raise
        else:
            train_cell, valid_cell = point_cell.partition(point_cell.train_cleaned_velo)

            train_point, train_meta, train_num, _, _, _ = self.extract_point(train_cell, level_idx)
            valid_point, valid_meta, valid_num, _, _, _ = self.extract_point(valid_cell, level_idx)
            test_point, test_meta, test_num, _, _, _ = self.extract_point(point_cell.test_cell, level_idx)
    
        return train_point, valid_point, test_point, train_num, valid_num, test_num

    def predict_test(self, point_cell, ckpt_name, mode):
        '''
        wrapper for evaluate_sweep function
        '''
        ckpt_path = util.create_dir(osp.join(self.save_path, 'level_%s' % (self.current_level)))
        # self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, os.path.join(ckpt_path, ckpt_name))
        self.evaluate_sweep(point_cell, compress=mode, num_points=(self.latent_dim//3))

    def predict(self, points, info, ckpt_name):

        meta = np.array([info[i]['num_points'] for i in range(len(info))])
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, os.path.join(self.save_path, ckpt_name))
        feed_dict = {self.pc: np.array(points), self.meta: meta, self.is_training: False}
        point_reconstr, reconstr_loss = self.sess.run([self.x_reconstr, self.reconstruction_loss], feed_dict)
        return point_reconstr

    def compress(self, point_cell, ckpt_name):
        '''
        compress one entire sweep data
        '''
        ckpt_path = util.create_dir(osp.join(self.save_path, 'level_%s' % (self.current_level)))
        # self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, os.path.join(ckpt_path, ckpt_name))
        self.compress_sweep(point_cell, num_points=(self.latent_dim//3))

    def compress_sweep(self, point_cell, num_points, level_idx=0, num_partition=4):
        all_sweep = point_cell.train_cleaned_velo + point_cell.test_cleaned_velo
        print ("compression in %d blocks" % num_partition)
        partition_size = len(all_sweep)//num_partition
        pred_code = []
        residual_code = []
        pred_all = []
        orig_all = []
        for idx_p in range(0, num_partition):
            print ("\nPartitioning...")
            sweep_s, sweep_e = idx_p*partition_size, max((idx_p+1)*partition_size, len(all_sweep))
            partition_sweep = all_sweep[sweep_s:sweep_e]
            points = (point_cell.partition_batch(partition_sweep, permutation=False))[0]
            if self.range_view == False:
                sweep_size = self.L[level_idx] * self.W[level_idx]
            else:
                sweep_size = point_cell.image_height[level_idx] * point_cell.image_width[level_idx]

            pred_code_block = []
            residual_code_block = []
            pred_all_block = []
            orig_all_block = []
            print ("\nCompression...\n")
            for i in tqdm(range(0, len(points), sweep_size)):
                s, e = i, i+sweep_size
                sweep_compress, compress_meta, sweep_orig, orig_meta = self.extract_sweep(points, s, e, level_idx)
                gt_points = point_cell.reconstruct_scene(sweep_compress, compress_meta)
                orig_points = point_cell.reconstruct_scene(sweep_orig, orig_meta)
                orig_sweep = np.concatenate([gt_points, orig_points], axis=0)

                compress_nums = compress_meta.as_matrix(columns=['num_points']).squeeze().astype(int)
                compress_row = compress_meta.as_matrix(columns=['row']).squeeze().astype(int)
                compress_col = compress_meta.as_matrix(columns=['col']).squeeze().astype(int)
                orig_nums = orig_meta.as_matrix(columns=['num_points']).squeeze().astype(int)
                total_compress_num = orig_nums.sum() + compress_nums.shape[0]*num_points
                total_points = []
                if self.range_view == False:
                    sweep_code = np.zeros([self.L[level_idx], self.W[level_idx], self.latent_dim])
                else:
                    sweep_code = np.zeros([point_cell.image_height[level_idx], point_cell.image_width[level_idx], self.latent_dim])

                test_batch_size = self.batch_size # hard coded, do not change other value except [64,128,256,512]
                for j in range(len(sweep_compress)//test_batch_size + 1):
                    valid_s, valid_e = j*test_batch_size, min((j+1)*test_batch_size, len(sweep_compress))
                    if valid_e - valid_s == test_batch_size:
                        fetched_pc = sweep_compress[valid_s:valid_e]
                        fetched_meta = compress_nums[valid_s:valid_e]
                    else:
                        pad_size = test_batch_size - (valid_e - valid_s)
                        pad_shape = [pad_size] + list(sweep_compress[valid_s:valid_e].shape[1:])
                        fetched_pc = np.concatenate([sweep_compress[valid_s:valid_e], np.zeros(pad_shape)], axis=0)
                        fetched_meta = np.concatenate([compress_nums[valid_s:valid_e], np.zeros(pad_size)], axis=0)
                    feed_dict = {self.pc: fetched_pc,
                                 self.meta: fetched_meta,
                                 self.is_training: False}
                    code, recon = self.sess.run([self.latent_code, self.x_reconstr], feed_dict)
                    r, c = compress_row[valid_s:valid_e], compress_col[valid_s:valid_e]
                    # emd loss and chamfer loss return mean of batch, should multiply batch_size
                    sweep_code[r,c] = code[:(valid_e-valid_s)]
                    total_points.append(recon)

                # orig = np.concatenate([sweep_compress[~np.all(sweep_compress == 0, axis=2)], sweep_orig[~np.all(sweep_orig == 0, axis=2)]], axis=0)
                total_points = np.concatenate(total_points, axis=0)
                pred_points = point_cell.reconstruct_scene(total_points, compress_meta)
                pred_all_block.append(np.concatenate([pred_points, orig_points], axis=0))
                orig_all_block.append(orig_sweep)
                pred_code_block.append(sweep_code)
                residual_code_block.append(orig_points)

            pred_all.extend(pred_all_block)
            orig_all.extend(orig_all_block)
            pred_code.extend(pred_code_block)
            residual_code.extend(residual_code_block)

        print ("Saving compression latent code")
        with h5py.File(self.save_path+'/code.h5', 'w', libver='latest') as f:
            for idx, arr in enumerate(pred_code):
                dset = f.create_dataset(str(idx), data=arr)

        print ("Saving uncompression points")
        with h5py.File(self.save_path+'/aux_code.h5', 'w', libver='latest') as f:
            for idx, arr in enumerate(residual_code):
                dset = f.create_dataset(str(idx), data=arr)

        print ("Saving reconstruction")
        with h5py.File(self.save_path+'/recon.h5', 'w', libver='latest') as f:
            for idx, arr in enumerate(pred_all):
                dset = f.create_dataset(str(idx), data=arr)
                        
        print ("Saving ground truth")
        with h5py.File(self.save_path+'/orig.h5', 'w', libver='latest') as f:
            for idx, arr in enumerate(orig_all):
                dset = f.create_dataset(str(idx), data=arr)


    # def reconstruct_sweep(self, ):
    #     '''
    #     reconstruct one entire sweep data
    #     '''



class StackAutoEncoder(AutoEncoder):
    def __init__(self, FLAGS):
        self.combination = FLAGS.combination
        self.table = {'down': 0, 'up': -1, 'sandwich': 1}
        self.label_level = self.table[self.combination]
        super(StackAutoEncoder, self).__init__(FLAGS)
        
    def _ae(self, pc, mask, bn_decay):

        label_level = self.level-1 if self.label_level==-1 else self.label_level
        
        # ------- encoder -------
        stacked_code = []
        for l in range(self.level): 
            net, rotation = self._encoder(pc[l], mask[l], bn_decay, name_scope='encoder_l%d'%l)

            # max pool
            if self.pooling == 'mean':
                code = tf.reduce_mean(net, axis=-2, keepdims=False)
            elif self.pooling == 'max':
                code = tf.reduce_max(net, axis=-2, keepdims=False)
            print (code)
            stacked_code.append(code)
            if l == label_level:
                transform = rotation

        stacked_code = tf.concat(stacked_code, axis=-1)
        combined_code = self._fuse(stacked_code)

        # ------- decoder ------- 
        x_reconstr = self._decoder(combined_code, mask[self.label_level], bn_decay, name_scope='decoder')

        if self.rotation == True:
            # rotate_matrix = tf.stop_gradient(tf.matrix_inverse(transform))
            rotate_matrix = tf.matrix_inverse(transform)

            return tf.matmul(x_reconstr, rotate_matrix), combined_code
        else:
            return x_reconstr, combined_code

    def _fuse(self, code):
        code = tf.expand_dims(code, -2)
        code = tf_util.point_conv(code, 
                         self.is_training,
                         n_filters=[32, self.latent_dim],
                         activation_function=None,
                         name="fuse")
        return code

    def _build_cell(self, g):

        with g.as_default():
            self.pc, self.mask, self.meta = [], [], []
            for l in range(self.level):
                # MLP: all_points [batch, 200, 2] -> MLP -> node_feature [batch, 200, 10]
                self.pc.append(tf.placeholder(tf.float32, [None, self.origin_num_points[l], 3]))
                meta = tf.placeholder(tf.int32, [None])
                self.meta.append(meta)
                mask = tf.sequence_mask(meta, maxlen=self.origin_num_points[l], dtype=tf.float32)
                self.mask.append(tf.expand_dims(mask, -1))

            self.is_training = tf.placeholder(tf.bool, shape=())
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)

            bn_decay = self.get_bn_decay(batch)

            def calculate_chamfer(recon, orig, num_points):
                x_reconstr, pc = tf.expand_dims(recon[:num_points], 0), tf.expand_dims(orig[:num_points], 0)
                cost_p1_p2, _, cost_p2_p1, _ = nn_distance(x_reconstr, pc)
                chamfer_return = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
                chamfer_return = tf.where(tf.is_nan(chamfer_return), 0., chamfer_return)
                return chamfer_return

            # reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            learning_rate = self.get_learning_rate(batch)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

            tower_grads = []
            latent_gpu = []
            pred_gpu = []
            total_emd_loss, total_chamfer_loss = [], []
            for i in range(self.num_gpus):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    with tf.device('/gpu:%d'%(i)), tf.name_scope('gpu_%d'%(i)) as scope:
                        pc_batch = [tf.slice(layer_pc, [i*self.device_batch_size,0,0], [self.device_batch_size,-1,-1]) for layer_pc in self.pc]
                        meta_batch = [tf.slice(layer_meta, [i*self.device_batch_size], [self.device_batch_size]) for layer_meta in self.meta]
                        mask_batch = [tf.slice(layer_mask, [i*self.device_batch_size,0,0], [self.device_batch_size,-1,-1]) for layer_mask in self.mask]

                        x_reconstr, code = self._ae(pc_batch, mask_batch, bn_decay)
                        
                        label = pc_batch[self.label_level]
                        match = approx_match(label, x_reconstr)
                        emd_loss = tf.reduce_mean(match_cost(label, x_reconstr, match))
            
                        # self.chamfer_loss = calculate_chamfer(self.x_reconstr, self.pc)
                        chamfer_loss = tf.reduce_mean(tf.map_fn(lambda x: calculate_chamfer(x[0], x[1], x[2]), 
                                                                     (x_reconstr, pc_batch[self.label_level], meta_batch[self.label_level]), dtype=tf.float32))
                        #  -------  loss + optimization  ------- 
                        if self.loss_type == "emd":
                            reconstruction_loss = emd_loss
                        elif self.loss_type == "chamfer":
                            reconstruction_loss = chamfer_loss

                        grads = optimizer.compute_gradients(reconstruction_loss)
                        tower_grads.append(grads)
                        pred_gpu.append(x_reconstr)
                        latent_gpu.append(code)
                        total_emd_loss.append(emd_loss)
                        total_chamfer_loss.append(chamfer_loss)
                        
            self.x_reconstr = tf.concat(pred_gpu, 0)
            self.latent_code = tf.concat(latent_gpu, 0)
            self.emd_loss = tf.reduce_mean(total_emd_loss)
            self.chamfer_loss = tf.reduce_mean(total_chamfer_loss)
            grads = average_gradients(tower_grads)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.apply_gradients(grads, global_step=batch)
            
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            self.sess = tf.Session(config=config)

            self.sess.run(tf.global_variables_initializer())
            # self.saver = tf.train.Saver(tf.trainable_variables() + tf.get_collection_ref("batch_norm_non_trainable_variables_co‌​llection"))
            self.saver = tf.train.Saver()


    def feed_data(self, X, meta, is_training):
        if self.combination == 'up':
            cell_num = [1]
            for l in range(1, self.level):
                if self.range_view == True:
                    factor_W, factor_L = self.image_width[0]//self.image_width[l], self.image_height[0]//self.image_height[l]
                else:
                    factor_W, factor_L = self.W[0]//self.W[l], self.L[0]//self.L[l]
                cell_num.append(factor_W * factor_L)
            cell_num = np.insert(np.cumsum(cell_num[::-1]), 0, values=0)
        elif self.combination == 'sandwich':
            if self.range_view == True:
                factor_W, factor_L = self.image_width[0]//self.image_width[1], self.image_height[0]//self.image_height[1]
            else:
                factor_W, factor_L = self.W[0]//self.W[1], self.L[0]//self.L[1]
            cell_num = [0,factor_W*factor_L,factor_W*factor_L+1,factor_W*factor_L+2]

        feed_dict={self.is_training: is_training}
        for i in range(self.level):
            if self.combination == 'down':
                feed_dict[self.pc[i]] = X[:,i,...]
                feed_dict[self.meta[i]] = meta[:,i]
            else:
                s, e = cell_num[i], cell_num[i+1]
                feed_dict[self.pc[i]] = np.reshape(X[:,s:e,...], [-1, X.shape[-2], X.shape[-1]])
                feed_dict[self.meta[i]] = np.reshape(meta[:,s:e], -1)
        return feed_dict

    def train(self, point_cell):

        level_idx = -2 if self.combination == 'sandwich' else self.label_level
        train_point, valid_point, test_point, \
        train_meta, valid_meta, test_meta = self.preprocess(point_cell)

        record_loss = []
        
        num_batches = train_point.shape[0] // self.batch_size
        for epoch in range(self.num_epochs):

            train_idx = np.arange(0, len(train_point))
            np.random.shuffle(train_idx)
            current_point = train_point[train_idx]
            current_meta = train_meta[train_idx]

            record_mean, record_var, record_mse = [], [], []

            start = time.time()
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx+1) * self.batch_size

                X = current_point[start_idx:end_idx]
                meta = current_meta[start_idx:end_idx]

                feed_dict = self.feed_data(X, meta, is_training=True)
                op = [self.train_op, 
                      self.emd_loss,
                      self.chamfer_loss,
                      self.x_reconstr]
                _, emd_loss, chamfer_loss, recon = self.sess.run(op, feed_dict)
                if self.loss_type == 'emd':
                    record_loss.append(emd_loss)
                elif self.loss_type == 'chamfer':
                    record_loss.append(chamfer_loss)
                mean, var, mse = util.dist_stat(recon, X[:,level_idx,:,:], meta[:,level_idx], self.origin_num_points[level_idx])
                record_mean.append(mean)
                record_var.append(var)
                record_mse.append(mse)

                compression_ratio = (self.batch_size*self.latent_dim) / (np.sum(meta[:,level_idx])*3.0)
                if batch_idx % 100 == 0:
                    print ("iteration/epoch: {}/{}, optimizing {} loss, emd loss: {}, chamfer loss: {}".format(
                        batch_idx, epoch, self.loss_type, emd_loss, chamfer_loss))
                    print ("mean: %.6f, var: %.6f, mse: %.6f, compression ratio: %.6f\n" % 
                                                                      (np.array(record_mean).mean(),
                                                                       np.array(record_var).mean(),
                                                                       np.array(record_mse).mean(),
                                                                       compression_ratio))

            end = time.time()
            print("epoch: {}, elapsed time: {}".format(epoch, end - start))

            if epoch % self.save_freq == 0:
                ckpt_path = util.create_dir(osp.join(self.save_path, 'level_%s' % (self.current_level)))
                if self.fb_split == True:
                    ckpt_name = osp.join(ckpt_path, 'model-%s-%s.ckpt' % (mode, str(epoch)))
                else:
                    ckpt_name =  osp.join(ckpt_path, 'model-%s.ckpt' % (str(epoch)))
                
                self.saver.save(self.sess, ckpt_name)

                # evaluation
                self.evaluate(valid_point, valid_meta, test=False)
                self.evaluate(test_point, test_meta, test=True)

                def visualize_sample(sample_points, sample_meta, sample_num, orig_points, orig_meta, filename=None):
                    
                    meta_nums = sample_meta['stacked_num'].str[level_idx].values.astype(int)
                    if (meta_nums >= self.cell_min_points[0]).all():

                        sample_generated, sample_emd, sample_chamfer = [], [], []
                        for idx in range(0, sample_points.shape[0], self.batch_size):
                            start_idx, end_idx = idx, min(idx+self.batch_size, sample_points.shape[0])
                            if end_idx-start_idx == self.batch_size:
                                fetched_pc = sample_points[start_idx:end_idx]
                                fetched_pmeta = sample_num[start_idx:end_idx]
                            else:
                                pad_size = self.batch_size - (end_idx - start_idx)
                                pad_shape = [pad_size] + list(sample_points[start_idx:end_idx].shape[1:])
                                fetched_pc = np.concatenate([sample_points[start_idx:end_idx], np.zeros(pad_shape)], axis=0)
                                pmeta_shape = [pad_size] + ([self.level] if self.combination == 'down' else list(sample_num.shape[1:]))
                                fetched_pmeta = np.concatenate([sample_num[start_idx:end_idx], np.zeros(pmeta_shape)], axis=0)

                            feed_dict = self.feed_data(fetched_pc, fetched_pmeta, is_training=False)

                            sg, se, sc = self.sess.run([self.x_reconstr, 
                                                        self.emd_loss, 
                                                        self.chamfer_loss], feed_dict)
                            sample_generated.append(sg)
                            sample_emd.append(se)
                            sample_chamfer.append(sc)
                        sample_emd, sample_chamfer = np.mean(sample_emd), np.mean(sample_chamfer)
                        sample_generated = np.concatenate(sample_generated, 0)

                    else:
                        print ("compressed points should above min points settings!")
                        raise

                    reconstruction_sample = point_cell.reconstruct_scene(sample_generated, sample_meta)
                    reconstruction_orig = point_cell.reconstruct_scene(orig_points[:,level_idx,...], orig_meta)
                    reconstruction = np.concatenate([reconstruction_sample, reconstruction_orig], 0)

                    if filename == None:
                        sweep = point_cell.test_sweep[point_cell.sample_idx]
                    elif filename == 'foreground':
                        sweep = point_cell.test_f_sweep
                    elif filename == 'background':
                        sweep = point_cell.test_b_sweep
                    
                    mean, var, mse = util.sweep_stat(reconstruction, sweep)
                    print ('sampled sweep emd loss: {}, chamfer loss: {}, MSE: {}'.format(sample_emd,
                                                                                          sample_chamfer,
                                                                                          mse))
                    print ('save reconstructed sweep')
                    full_filename = 'orig_'+str(epoch) if filename == None else 'orig_{}_{}'.format(filename, str(epoch))
                    util.visualize_3d_points(sweep, dir=ckpt_path, filename=full_filename)
                    full_filename = 'recon_'+str(epoch) if filename == None else 'recon_{}_{}'.format(filename, str(epoch))
                    util.visualize_3d_points(reconstruction, dir=ckpt_path, filename=full_filename)
                    return reconstruction, sweep
                
                # plot
                meta_columns = ['index', 'ratio', 'center', 'level', 'parent_nump', 'num_points'] 

                sample_points, sample_meta, sample_num, orig_points, orig_meta, orig_num = self.extract_point(point_cell.sample_points)
                visualize_sample(sample_points, sample_meta, sample_num, orig_points, orig_meta)

        # check testing accuracy
        self.evaluate(test_point, test_meta, test=True)
        
    def evaluate(self, valid_point, valid_meta, test=True):
        
        level_idx = -2 if self.combination == 'sandwich' else self.label_level
        valid_emd_loss, valid_chamfer_loss = [], []
        i = 0
        mean_arr, var_arr, mse_arr = [], [], []
        for i in range(len(valid_point)//self.batch_size):
            valid_s, valid_e = i*self.batch_size, (i+1)*self.batch_size
            feed_dict = self.feed_data(valid_point[valid_s:valid_e], valid_meta[valid_s:valid_e], is_training=False)

            vel, vcl, recon = self.sess.run([self.emd_loss, self.chamfer_loss, self.x_reconstr], feed_dict)
            mean, var, mse = util.dist_stat(recon, 
                                            valid_point[valid_s:valid_e,level_idx,...], 
                                            valid_meta[valid_s:valid_e,level_idx], self.origin_num_points[level_idx])
            mean_arr.append(mean)
            var_arr.append(var)
            mse_arr.append(mse)
            valid_emd_loss.append(vel)
            valid_chamfer_loss.append(vcl)

        if test == True:
            print ('Testing emd loss: {}, Testing chamfer loss: {}'.format(np.array(valid_emd_loss).mean(),
                                                                           np.array(valid_chamfer_loss).mean()))
            print ('mean: %.6f, var: %.6f, mse: %.6f' % (np.array(mean_arr).mean(), 
                                                         np.array(var_arr).mean(), 
                                                         np.array(mse_arr).mean()))
        else:
            print ('validation emd loss: {}, validation chamfer loss: {}'.format(np.array(valid_emd_loss).mean(),
                                                                                 np.array(valid_chamfer_loss).mean()))
            print ('mean: %.6f, var: %.6f, mse: %.6f' % (np.array(mean_arr).mean(), 
                                                         np.array(var_arr).mean(), 
                                                         np.array(mse_arr).mean()))

    def preprocess(self, point_cell):
        train_cell, valid_cell = point_cell.partition(point_cell.train_cleaned_velo)
        train_point, train_meta, train_num, train_orig_point, train_orig_meta, train_orig_num = self.extract_point(train_cell)
        valid_point, valid_meta, valid_num, valid_orig_point, valid_orig_meta, valid_orig_num = self.extract_point(valid_cell)
        test_point, test_meta, test_num, test_orig_point, test_orig_meta, test_orig_num = self.extract_point(point_cell.test_cell)

        return train_point, valid_point, test_point, train_num, valid_num, test_num

    def group_multi_down(self, points, level_idx):
        print ("grouping multi scale cells...")
        if self.range_view == True:
            total_sweep_size = self.image_height[level_idx]*self.image_width[level_idx]
        else:
            total_sweep_size = self.L[level_idx]*self.W[level_idx]
        num_sweeps = int(points[level_idx].shape[0] / total_sweep_size)
        stacked_points = [np.array(list(points[level_idx].as_matrix(columns=['stacked']).squeeze()))]
        stacked_num = [np.array(list(points[level_idx].as_matrix(columns=['num_points']).squeeze()))] 
        for l_idx in range(1+level_idx, self.level):
            if self.range_view == False:
                factor_L = (self.L[level_idx]//self.L[l_idx])
                factor_W = (self.W[level_idx]//self.W[l_idx])
                sweep_size = self.L[l_idx]*self.W[l_idx]
                level_points = np.array(list(points[l_idx].as_matrix(columns=['points']).squeeze()))
                level_shape = [num_sweeps, self.L[l_idx], self.W[l_idx], level_points.shape[-2], 3]
            else:
                factor_L = (self.image_height[level_idx]//self.image_height[l_idx])
                factor_W = (self.image_width[level_idx]//self.image_width[l_idx])
                sweep_size = self.image_height[l_idx]*self.image_width[l_idx]
                level_points = np.array(list(points[l_idx].as_matrix(columns=['points']).squeeze()))
                level_shape = [num_sweeps, self.image_height[l_idx], self.image_width[l_idx], level_points.shape[-2], 3]
            level_points = np.reshape(level_points, level_shape)
            level_points = np.repeat(np.repeat(level_points, factor_L, axis=1), factor_W, axis=2)
            stacked_points.append(np.reshape(level_points, [-1, level_shape[-2], 3]))

            level_num = np.array(list(points[l_idx].as_matrix(columns=['num_points']).squeeze()))
            if self.range_view == False:
                level_num = np.reshape(level_num, [num_sweeps, self.L[l_idx], self.W[l_idx]])
            else:
                level_num = np.reshape(level_num, [num_sweeps, self.image_height[l_idx], self.image_width[l_idx]])
            level_num = np.repeat(np.repeat(level_num, factor_L, axis=1), factor_W, axis=2)
            stacked_num.append(np.reshape(level_num, -1))
        stacked_points = np.stack(stacked_points)
        stacked_num = np.stack(stacked_num)

        p = points[level_idx].copy()
        for i in range(p.shape[0]): # change -1 to 0
            p.iloc[i]['stacked'] = stacked_points[:,i,...] # change -1 to 0
            p.iloc[i]['stacked_num'] = stacked_num[:,i] # change -1 to 0

        return p

    def group_multi_middle(self, points):
        p_up = self.group_multi_up(points, level_idx=1)
        p_down = self.group_multi_down(points, level_idx=1)

        for i in range(p_up.shape[0]): # change -1 to 0
            stacked_p = np.concatenate([p_up.iloc[i]['stacked'], np.expand_dims(p_down.iloc[i]['stacked'][1,...], 0)], axis=0)
            stacked_num = np.concatenate([p_up.iloc[i]['stacked_num'], p_down.iloc[i]['stacked_num'][1:]], axis=0)
            p_up.iloc[i]['stacked'] = stacked_p
            p_up.iloc[i]['stacked_num'] = stacked_num

        return p_up

    def group_multi_up(self, points, level_idx):
        print ("grouping multi scale cells...")
        level_idx = self.level-1 if level_idx==-1 else level_idx
        if self.range_view == True:
            total_sweep_size = self.image_height[level_idx]*self.image_width[level_idx]
        else:
            total_sweep_size = self.L[level_idx]*self.W[level_idx]
        num_sweeps = int(points[level_idx].shape[0] / total_sweep_size)
        stacked_points, stacked_num = [], []
        for l_idx in range(1, level_idx+1):
            if self.range_view == False:
                factor_L = (self.L[l_idx-1]//self.L[level_idx])
                factor_W = (self.W[l_idx-1]//self.W[level_idx])
                sweep_size = self.L[l_idx-1]*self.W[l_idx-1]
                level_points = np.array(list(points[l_idx-1].as_matrix(columns=['points']).squeeze()))
                level_shape = [num_sweeps, self.L[l_idx-1], self.W[l_idx-1], level_points.shape[-2], 3]
            else:
                factor_L = (self.image_height[l_idx-1]//self.image_height[level_idx])
                factor_W = (self.image_width[l_idx-1]//self.image_width[level_idx])
                sweep_size = self.image_height[l_idx-1]*self.image_width[l_idx-1]
                level_points = np.array(list(points[l_idx].as_matrix(columns=['points']).squeeze()))
                level_shape = [num_sweeps, self.image_height[l_idx-1], self.image_width[l_idx-1], level_points.shape[-2], 3]
            level_points = np.reshape(level_points, level_shape)
            level_num = np.array(list(points[l_idx-1].as_matrix(columns=['num_points']).squeeze()))
            if self.range_view == False:
                level_num = np.reshape(level_num, [num_sweeps, self.L[l_idx-1], self.W[l_idx-1]])
            else:
                level_num = np.reshape(level_num, [num_sweeps, self.image_height[l_idx-1], self.image_width[l_idx-1]])

            buf_points, buf_num = [], []
            for i in range(factor_L):
                for j in range(factor_W):
                    buf_points.append(level_points[:,i::factor_L,j::factor_W,...])
                    buf_num.append(level_num[:,i::factor_L,j::factor_W])
            packed_points = np.reshape(np.transpose(np.stack(buf_points), [1,2,3,0,4,5]), [-1, factor_L*factor_W, level_shape[-2], 3])
            stacked_points.append(packed_points)
            packed_num = np.reshape(np.transpose(np.stack(buf_num), [1,2,3,0]), [-1, factor_L*factor_W])
            stacked_num.append(packed_num)
        
        stacked_points.append(np.expand_dims(np.array(list(points[level_idx].as_matrix(columns=['stacked']).squeeze())), axis=1))
        stacked_num.append(np.expand_dims(np.array(list(points[level_idx].as_matrix(columns=['num_points']).squeeze())), axis=1))
        stacked_points = np.concatenate(stacked_points, axis=1)
        stacked_num = np.concatenate(stacked_num, axis=1)

        p = points[level_idx].copy()
        for i in range(p.shape[0]): # change -1 to 0
            p.iloc[i]['stacked'] = stacked_points[i,...] # change -1 to 0
            p.iloc[i]['stacked_num'] = stacked_num[i,...] # change -1 to 0

        return p

    def extract_point(self, points):
        if self.combination == 'down':
            level_idx=0
            point_group = self.group_multi_down(points, level_idx)
        elif self.combination == 'sandwich':
            point_group = self.group_multi_middle(points)
            level_idx=-2
        else:
            level_idx=-1
            point_group = self.group_multi_up(points, level_idx)
        low_points = point_group
        sample_points_df = low_points[low_points['stacked_num'].str[level_idx] >= self.cell_min_points[level_idx]]
        stacked_points = sample_points_df.as_matrix(columns=['stacked'])
        stacked_points = np.array(list(stacked_points.squeeze()))
        sample_num = np.array(list(sample_points_df.as_matrix(columns=['stacked_num']).squeeze())).astype(int)
        sample_meta = sample_points_df

        orig_points_df = low_points[(low_points['stacked_num'].str[level_idx] < self.cell_min_points[level_idx]) & (low_points['stacked_num'].str[level_idx] > 0)]
        orig_points = orig_points_df.as_matrix(columns=['stacked'])
        orig_points = np.array(list(orig_points.squeeze()))
        orig_num = np.array(list(orig_points_df.as_matrix(columns=['stacked_num']).squeeze())).astype(int)
        orig_meta = orig_points_df

        return stacked_points, sample_meta, sample_num, orig_points, orig_meta, orig_num

    def extract_sweep(self, points, start_idx, end_idx, level_idx=0):
        low_points = points
        level_points = low_points.iloc[start_idx:end_idx]
        sample_points_df = level_points[level_points['stacked_num'].str[level_idx] >= self.cell_min_points[level_idx]]
        sample_points = sample_points_df.as_matrix(columns=['stacked'])
        sample_points = np.array(list(sample_points.squeeze()))
        sample_num = np.array(list(sample_points_df.as_matrix(columns=['stacked_num']).squeeze())).astype(int)
        sample_meta = sample_points_df

        orig_points_df = level_points[(level_points['stacked_num'].str[level_idx] < self.cell_min_points[level_idx]) & (level_points['stacked_num'].str[level_idx] > 0)]
        orig_points = orig_points_df.as_matrix(columns=['stacked'])
        orig_points = np.array(list(orig_points.squeeze()))
        orig_num = orig_points_df.as_matrix(columns=['num_points']).squeeze().astype(int)
        orig_meta = orig_points_df

        return sample_points, sample_meta, sample_num, orig_points, orig_meta, orig_num

