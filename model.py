import numpy as np
import tensorflow as tf
import util
import os
import os.path as osp
import sys
import collections

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util
import provider

from external.structural_losses.tf_nndistance import nn_distance
from external.structural_losses.tf_approxmatch import approx_match, match_cost


class AutoEncoder():
    def __init__(self, FLAGS, current_level):
        self.batch_size = FLAGS.batch_size        
        self.latent_dim = FLAGS.latent_dim
        self.num_epochs = FLAGS.num_epochs
        self.save_freq = FLAGS.save_freq
        self.save_path = FLAGS.save_path
        self.fb_split = FLAGS.fb_split
        self.level = len(FLAGS.PW)

        self.learning_rate = FLAGS.learning_rate
        self.loss_type = FLAGS.loss_type
        self.origin_num_points = list(map(int, FLAGS.cell_max_points.split(',')))
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
        self.current_level = current_level

        self.g = tf.Graph()
        self._build_cell(self.g)

    def _encoder(self, net, name_scope):
        with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
            net = tf_util.point_conv(input_image=net, 
                                     is_training=self.is_training, 
                                     n_filters=[64, 64], 
                                     name="encoder1")
            net = tf_util.point_conv(input_image=net, 
                                     is_training=self.is_training, 
                                     n_filters=[64, self.latent_dim],
                                     name="encoder2")

        return self.mask * net

    def _decoder(self, code, name_scope):
        with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
            if self.decoder_mode == 'pointnet':
                x_reconstr = tf_util.fully_connected_decoder(code, 
                                                             self.origin_num_points[self.current_level], 
                                                             self.is_training)
            elif self.decoder_mode == 'pointgrid':
                code = tf.expand_dims(code, -2)
                global_code_tile = tf.tile(code, [1, self.origin_num_points[self.current_level], 1])
                local_code_tile = tf_util.fold3d(self.origin_num_points[self.current_level], 
                                                 self.map_size,
                                                 tf.shape(code)[0])
                all_code = tf.concat([local_code_tile, global_code_tile], axis=-1)

                net = tf_util.point_conv(all_code, 
                                         self.is_training, 
                                         n_filters=[256, 128, 64], 
                                         activation_function=tf.nn.relu,
                                         name="decoder1")
                net = tf_util.point_conv(net, 
                                         self.is_training, 
                                         n_filters=[3],
                                         activation_function=None,
                                         name="decoder2")
                # second folding
                if self.enable_2ndfold:
                    net = tf.concat([local_code_tile, net], axis=-1)
                    net = tf_util.point_conv(net, 
                                             self.is_training, 
                                             n_filters=[258, 128, 64], 
                                             activation_function=tf.nn.relu,
                                             name="decoder3")
                    net = tf_util.point_conv(net, 
                                             self.is_training, 
                                             n_filters=[3],
                                             activation_function=None,
                                             name="decoder4")
                net = tf.reshape(net, [-1, self.origin_num_points[self.current_level], 3])
                net_xy = tf.nn.tanh(net[:,:,0:2])
                net_z = tf.expand_dims(net[:,:,2], -1)
                x_reconstr = tf.concat([net_xy, net_z], -1)
        return self.mask * x_reconstr

    def _ae(self, pc):

        # ------- encoder -------
        net = self._encoder(pc, name_scope='encoder')
        
        # max pool
        code = tf.reduce_max(net, axis=-2, keepdims=False)
        print (code)

        # ------- decoder ------- 
        x_reconstr = self._decoder(code, name_scope='decoder')

        return x_reconstr

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

            self.x_reconstr = self._ae(self.pc)

            match = approx_match(self.x_reconstr, self.pc)
            self.emd_loss = tf.reduce_mean(match_cost(self.x_reconstr, self.pc, match))

            def calculate_chamfer(recon, orig, num_points):
                x_reconstr, pc = tf.expand_dims(recon[:num_points], 0), tf.expand_dims(orig[:num_points], 0)
                cost_p1_p2, _, cost_p2_p1, _ = nn_distance(x_reconstr, pc)
                return tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)

            # self.chamfer_loss = calculate_chamfer(self.x_reconstr, self.pc)
            self.chamfer_loss = tf.reduce_mean(tf.map_fn(lambda x: calculate_chamfer(x[0], x[1], x[2]), 
                                                         (self.x_reconstr, self.pc, self.meta), dtype=tf.float32))
            #  -------  loss + optimization  ------- 
            if self.loss_type == "emd":
                self.reconstruction_loss = self.emd_loss
            elif self.loss_type == "chamfer":
                self.reconstruction_loss = self.chamfer_loss

            # reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            learning_rate = self.get_learning_rate(batch)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.reconstruction_loss, global_step=batch)
            
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            self.sess = tf.Session(config=config)

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

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
        for epoch in range(self.num_epochs):

            train_idx = np.arange(0, len(dataset.train_set))
            np.random.shuffle(train_idx)
            current_point = train_point[train_idx]
            current_meta = train_meta[train_idx]

            record_mean, record_var, record_mse = [], [], []
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
            if epoch % self.save_freq == 0:
                ckpt_path = util.create_dir(osp.join(self.save_path, 'level_%s' % (self.current_level)))
                if self.fb_split == True:
                    ckpt_name = osp.join(ckpt_path, 'model-%s-%s.ckpt' % (mode, str(epoch)))
                else:
                    ckpt_name =  osp.join(ckpt_path, 'model-%s.ckpt' % (str(epoch)))
                
                self.saver.save(self.sess, ckpt_name)

                # evaluation
                valid_emd_loss, valid_chamfer_loss = [], []
                i = 0
                for i in range(len(valid_point)//self.batch_size):
                    valid_s, valid_e = i*self.batch_size, (i+1)*self.batch_size
                    feed_dict = {self.pc: valid_point[valid_s:valid_e],
                                 self.meta: valid_meta[valid_s:valid_e],
                                 self.is_training: False}
                    vel, vcl = self.sess.run([self.emd_loss, self.chamfer_loss], feed_dict)
                    valid_emd_loss.append(vel)
                    valid_chamfer_loss.append(vcl)

                feed_dict = {self.pc: valid_point[i*self.batch_size:],
                             self.meta: valid_meta[i*self.batch_size:], 
                             self.is_training: False}
                vel, vcl = self.sess.run([self.emd_loss, self.chamfer_loss], feed_dict)
                valid_emd_loss.append(vel)
                valid_chamfer_loss.append(vcl)
                print ('validation emd loss: {}, validation chamfer loss: {}'.format(np.array(valid_emd_loss).mean(),
                                                                                         np.array(valid_chamfer_loss).mean()))

                def visualize_sample(sample_points, sample_meta, filename=None):
                    meta_nums = sample_meta['num_points'].values.astype(int)
                    feed_dict = {self.pc: sample_points, 
                                 self.meta: meta_nums, 
                                 self.is_training: False}
                    sample_generated, sample_emd, sample_chamfer = self.sess.run([self.x_reconstr, 
                                                                   self.emd_loss, 
                                                                   self.chamfer_loss], feed_dict)
                    reconstruction = point_cell.reconstruct_scene(sample_generated, sample_meta)
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
                meta_columns = ['index', 'ratio', 'center', 'level', 'num_points'] 
                if self.fb_split == True:
                    sample_points = point_cell.sample_points_f[level_idx].as_matrix(columns=['points'])
                    sample_points = np.array(list(sample_points.squeeze()))
                    sample_meta = point_cell.sample_points_f[level_idx][meta_columns]
                    rf, sf = visualize_sample(sample_points, sample_meta, 'foreground')

                    sample_points = point_cell.sample_points_b[level_idx].as_matrix(columns=['points'])
                    sample_points = np.array(list(sample_points.squeeze()))
                    sample_meta = point_cell.sample_points_b[level_idx][meta_columns]
                    rb, sb = visualize_sample(sample_points, sample_meta, 'background')
                    reconstruction = np.concatenate([rf, rb], 0)
                    sweep = np.concatenate([sf, sb], 0)
                    util.visualize_3d_points(sweep, dir=ckpt_path, filename='orig_{}'.format(str(epoch)))
                    util.visualize_3d_points(reconstruction, dir=ckpt_path, filename='recon_{}'.format(str(epoch)))
                else:
                    sample_points = point_cell.sample_points[level_idx].as_matrix(columns=['points'])
                    sample_points = np.array(list(sample_points.squeeze()))
                    sample_meta = point_cell.sample_points[level_idx][meta_columns]                  
                    visualize_sample(sample_points, sample_meta)

        # check testing accuracy
        test_emd_loss, test_chamfer_loss = [], []
        i = 0
        for i in range(len(test_point)//self.batch_size):
            test_s, test_e = i*self.batch_size, (i+1)*self.batch_size
            feed_dict = {self.pc: test_point[test_s:test_e],
                         self.meta: test_meta[test_s:test_e],
                         self.is_training: False}
            vel, vcl = self.sess.run([self.emd_loss, self.chamfer_loss], feed_dict)
            test_emd_loss.append(vel)
            test_chamfer_loss.append(vcl)

        feed_dict = {self.pc: test_point[i*self.batch_size:],
                     self.meta: test_meta[i*self.batch_size:], 
                     self.is_training: False}
        vel, vcl = self.sess.run([self.emd_loss, self.chamfer_loss], feed_dict)
        test_emd_loss.append(vel)
        test_chamfer_loss.append(vcl)
        print ('validation emd loss: {}, validation chamfer loss: {}'.format(np.array(test_emd_loss).mean(),
                                                                             np.array(test_chamfer_loss).mean()))


                
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


    def preprocess(self, point_cell, level_idx, mode='foreground'):
        if self.fb_split == True:
            if mode == 'foreground':
                train_point = point_cell.train_cell_f[level_idx].as_matrix(columns=['points'])
                train_point = np.array(list(train_point.squeeze()))
                valid_point = point_cell.valid_cell_f[level_idx].as_matrix(columns=['points'])
                valid_point = np.array(list(valid_point.squeeze()))
                test_point = point_cell.test_cell_f[level_idx].as_matrix(columns=['points'])
                test_point = np.array(list(test_point.squeeze()))

                train_meta = point_cell.train_cell_f[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)
                valid_meta = point_cell.valid_cell_f[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)
                test_meta = point_cell.test_cell_f[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)

            elif mode == 'background':
                train_point = point_cell.train_cell_b[level_idx].as_matrix(columns=['points'])
                train_point = np.array(list(train_point.squeeze()))
                valid_point = point_cell.valid_cell_b[level_idx].as_matrix(columns=['points'])
                valid_point = np.array(list(valid_point.squeeze()))
                test_point = point_cell.test_cell_b[level_idx].as_matrix(columns=['points'])
                test_point = np.array(list(test_point.squeeze()))

                train_meta = point_cell.train_cell_b[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)
                valid_meta = point_cell.valid_cell_b[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)
                test_meta = point_cell.test_cell_b[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)

            else:
                print ("unkonwn mode {}, should be either foreground or background!" % mode)
                raise
        else:
            train_point = point_cell.train_cell[level_idx].as_matrix(columns=['points'])
            train_point = np.array(list(train_point.squeeze()))
            valid_point = point_cell.valid_cell[level_idx].as_matrix(columns=['points'])
            valid_point = np.array(list(valid_point.squeeze()))
            test_point = point_cell.test_cell[level_idx].as_matrix(columns=['points'])
            test_point = np.array(list(test_point.squeeze()))
        
            train_meta = point_cell.train_cell[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)
            valid_meta = point_cell.valid_cell[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)
            test_meta = point_cell.test_cell[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)

        return train_point, valid_point, test_point, train_meta, valid_meta, test_meta

    def predict(self, points, info, ckpt_name, mode='foreground'):

        meta = np.array([info[i]['num_points'] for i in range(len(info))])
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, os.path.join(self.save_path, ckpt_name))
        feed_dict = {self.pc: np.array(points), self.meta: meta, self.is_training: False}
        point_reconstr, reconstr_loss = self.sess.run([self.x_reconstr, self.reconstruction_loss], feed_dict)
        return point_reconstr

    # def compress_cell(self, pc_data):
    #     '''
    #     compress a batch of cell
    #     '''
        

    # def reconstruct_cell(self, ):
    #     '''
    #     reconstruct a batch of cell
    #     '''


    # def compress_sweep(self, ):
    #     '''
    #     compress one entire sweep data
    #     '''
        

    # def reconstruct_sweep(self, ):
    #     '''
    #     reconstruct one entire sweep data
    #     '''



# class StackAutoEncoder():
#     def __init__(self, FLAGS):
#         self.batch_size = FLAGS.batch_size        
#         self.latent_dim = FLAGS.latent_dim
#         self.num_epochs = FLAGS.num_epochs
#         self.save_freq = FLAGS.save_freq
#         self.save_path = FLAGS.save_path
#         self.fb_split = FLAGS.fb_split
#         self.level = len(FLAGS.PW)

#         self.factor = FLAGS.PL[1] / FLAGS.PL[0] if len(FLAGS.PL) > 1 else 1
#         self.learning_rate = FLAGS.learning_rate
#         self.loss_type = FLAGS.loss_type
#         self.origin_num_points = list(map(int, FLAGS.cell_max_points.split(',')))
#         self.DECAY_RATE = 0.8
#         self.DECAY_STEP = FLAGS.decay_step
#         self.BN_INIT_DECAY = 0.5
#         self.BN_DECAY_DECAY_RATE = 0.5
#         self.BN_DECAY_DECAY_STEP = float(self.DECAY_STEP)
#         self.BN_DECAY_CLIP = 0.99
#         self.enable_2ndfold = True
#         self.map_size = [8, 8, 8]
#         self.encoder_mode = FLAGS.encoder
#         self.decoder_mode = FLAGS.decoder

#         self.g = tf.Graph()
#         self._build_model(self.g)

#     def _encoder(self, net, mask, name_scope):
#         with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
#             net = tf_util.point_conv(input_image=net, 
#                                      is_training=self.is_training, 
#                                      n_filters=[64, 64], 
#                                      name="encoder1")
#             net = tf_util.point_conv(input_image=net, 
#                                      is_training=self.is_training, 
#                                      n_filters=[64, self.latent_dim],
#                                      name="encoder2")

#         return mask * net

#     def _decoder(self, code, name_scope):
#         with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
#             if self.decoder_mode == 'pointnet':
#                 x_reconstr = tf_util.fully_connected_decoder(code, 
#                                                              self.origin_num_points[1], 
#                                                              self.is_training)
#             elif self.decoder_mode == 'pointgrid':
#                 code = tf.expand_dims(code, -2)
#                 global_code_tile = tf.tile(code, [1, self.origin_num_points[1], 1])
#                 local_code_tile = tf_util.fold3d(self.origin_num_points[1], 
#                                                  self.map_size,
#                                                  tf.shape(code)[0])
#                 all_code = tf.concat([local_code_tile, global_code_tile], axis=-1)

#                 net = tf_util.point_conv(all_code, 
#                                          self.is_training, 
#                                          n_filters=[256, 128, 64], 
#                                          activation_function=tf.nn.relu,
#                                          name="decoder1")
#                 net = tf_util.point_conv(net, 
#                                          self.is_training, 
#                                          n_filters=[3],
#                                          activation_function=None,
#                                          name="decoder2")
#                 # second folding
#                 if self.enable_2ndfold:
#                     net = tf.concat([local_code_tile, net], axis=-1)
#                     net = tf_util.point_conv(net, 
#                                              self.is_training, 
#                                              n_filters=[258, 128, 64], 
#                                              activation_function=tf.nn.relu,
#                                              name="decoder3")
#                     net = tf_util.point_conv(net, 
#                                              self.is_training, 
#                                              n_filters=[3],
#                                              activation_function=None,
#                                              name="decoder4")
#                 net = tf.reshape(net, [-1, self.origin_num_points[1], 3])
#                 net_xy = tf.nn.tanh(net[:,:,0:2])
#                 net_z = tf.expand_dims(net[:,:,2], -1)
#                 x_reconstr = tf.concat([net_xy, net_z], -1)
#         return self.mask * x_reconstr

#     def _fuse(self, code):
#         code = tf.expand_dims(code, -2)
#         code = tf_util.point_conv(code, 
#                          self.is_training, 
#                          n_filters=[32, self.latent_dim],
#                          activation_function=None,
#                          name="fuse")
#         return code

#     def _ae(self, top_pc, top_mask, low_pc, low_mask):

#         # ------- encoder -------
#         top_latent = self._encoder(top_pc, top_mask, name_scope='top_encoder')
#         top_code = tf.reduce_max(top_latent, axis=-2, keepdims=False)

#         if len(self.origin_num_points) > 1:
#             low_code = []
#             for l in range(self.factor):
#                 # ------- encoder -------
#                 low_latent = self._encoder(low_pc[:, l], low_mask[:, l], name_scope='low_encoder')
#                 low_code.append(tf.reduce_max(low_latent, axis=-2, keepdims=False))

#             l_code = tf.stack(low_code, axis=1)
#             t_code = tf.tile(tf.expand_dims(top_code, axis=1), [1, self.factor, 1])
#             code = tf.concat([l_code, t_code], axis=-1)

#             combined_code = self._fuse(code)
#         else:
#             combined_code = top_code

#         # ------- decoder ------- 
#         x_reconstr = self._decoder(combined_code, name_scope='decoder')

#         return x_reconstr

#     def _build_model(self, g):
#         with g.as_default():
#             # MLP: all_points [batch, 200, 2] -> MLP -> node_feature [batch, 200, 10]
#             self.top_pc = tf.placeholder(tf.float32, [None, self.origin_num_points[0], 3])
#             self.top_meta = tf.placeholder(tf.int32, [None])
#             top_mask = tf.sequence_mask(self.top_meta, maxlen=self.origin_num_points[0], dtype=tf.float32)
#             self.top_mask = tf.expand_dims(top_mask, -1)

#             if len(self.origin_num_points) > 1:
#                 self.low_pc = tf.placeholder(tf.float32, [None, self.factor, self.origin_num_points[1], 3])
#                 self.low_meta = tf.placeholder(tf.int32, [None, self.factor])
#                 low_mask = []
#                 for i in range(self.factor):
#                     low_mask.append(tf.sequence_mask(self.low_meta[:,i], maxlen=self.origin_num_points[1], dtype=tf.float32))
#                 self.low_mask = tf.expand_dims(tf.stack(low_mask, axis=1), -1)

#             self.is_training = tf.placeholder(tf.bool, shape=())
            
#             batch = tf.get_variable('batch', [],
#                 initializer=tf.constant_initializer(0), trainable=False)

#             self.x_reconstr = self._ae(self.top_pc, self.top_mask, self.low_pc, self.low_mask)

#             match = approx_match(self.x_reconstr, self.top_pc)
#             self.emd_loss = tf.reduce_mean(match_cost(self.x_reconstr, self.top_pc, match))

#             def calculate_chamfer(recon, orig, num_points):
#                 x_reconstr, pc = tf.expand_dims(recon[:num_points], 0), tf.expand_dims(orig[:num_points], 0)
#                 cost_p1_p2, _, cost_p2_p1, _ = nn_distance(x_reconstr, pc)
#                 return tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)

#             # self.chamfer_loss = calculate_chamfer(self.x_reconstr, self.pc)
#             self.chamfer_loss = tf.reduce_mean(tf.map_fn(lambda x: calculate_chamfer(x[0], x[1], x[2]), 
#                                                          (self.x_reconstr, self.top_pc, self.top_meta), dtype=tf.float32))
#             #  -------  loss + optimization  ------- 
#             if self.loss_type == "emd":
#                 self.reconstruction_loss = self.emd_loss
#             elif self.loss_type == "chamfer":
#                 self.reconstruction_loss = self.chamfer_loss

#             # reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

#             learning_rate = self.get_learning_rate(batch)
#             optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

#             update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#             with tf.control_dependencies(update_ops):
#                 self.train_op = optimizer.minimize(self.reconstruction_loss, global_step=batch)
            
#             config = tf.ConfigProto()
#             config.gpu_options.allow_growth = True
#             config.allow_soft_placement = True
#             config.log_device_placement = False
#             self.sess = tf.Session(config=config)

#             self.sess.run(tf.global_variables_initializer())
#             self.saver = tf.train.Saver()

#     def train(self, point_cell, mode='foreground'):
        
#         train_point, valid_point, test_point, \
#         train_meta, valid_meta, test_meta = self.preprocess(point_cell, self.current_level, mode)

#         record_loss = []

#         num_batches = train_point.shape[0] // self.batch_size
#         for epoch in range(self.num_epochs):

#             train_idx = np.arange(0, len(dataset.train_set))
#             np.random.shuffle(train_idx)
#             current_point = train_point[train_idx]
#             current_meta = train_meta[train_idx]

#             record_mean, record_var, record_mse = [], [], []
#             for batch_idx in range(num_batches):
#                 start_idx = batch_idx * self.batch_size
#                 end_idx = (batch_idx+1) * self.batch_size

#                 X = current_point[start_idx:end_idx]
#                 meta = current_meta[start_idx:end_idx]
#                 if len(self.origin_num_points) > 1:
#                     feed_dict={self.top_pc: X, 
#                                # self.low_pc: ,
#                                self.top_meta: meta, 
#                                # self.low_meta: ,
#                                self.is_training: True}
#                 else:
#                     feed_dict={self.top_pc: X, self.top_meta: meta, self.is_training: True}
#                 op = [self.train_op, 
#                       self.emd_loss,
#                       self.chamfer_loss,
#                       self.x_reconstr,]
#                 _, emd_loss, chamfer_loss, recon = self.sess.run(op, feed_dict)
#                 if self.loss_type == 'emd':
#                     record_loss.append(emd_loss)
#                 elif self.loss_type == 'chamfer':
#                     record_loss.append(chamfer_loss)
#                 mean, var, mse = util.dist_stat(recon, X, meta, self.origin_num_points[level_idx])
#                 record_mean.append(mean)
#                 record_var.append(var)
#                 record_mse.append(mse)

#                 compression_ratio = (self.batch_size*self.latent_dim) / (np.sum(meta)*3.0)
#                 if batch_idx % 100 == 0:
#                     print ("iteration/epoch: {}/{}, optimizing {} loss, emd loss: {}, chamfer loss: {}".format(
#                         batch_idx, epoch, self.loss_type, emd_loss, chamfer_loss))
#                     print ("mean: %.6f, var: %.6f, mse: %.6f, compression ratio: %.6f\n" % 
#                                                                       (np.array(record_mean).mean(), 
#                                                                        np.array(record_var).mean(), 
#                                                                        np.array(record_mse).mean(),
#                                                                        compression_ratio))
#             if epoch % self.save_freq == 0:
#                 ckpt_path = util.create_dir(osp.join(self.save_path, 'level_%s' % (self.current_level)))
#                 if self.fb_split == True:
#                     ckpt_name = osp.join(ckpt_path, 'model-%s-%s.ckpt' % (mode, str(epoch)))
#                 else:
#                     ckpt_name =  osp.join(ckpt_path, 'model-%s.ckpt' % (str(epoch)))
                
#                 self.saver.save(self.sess, ckpt_name)

#                 # evaluation
#                 valid_emd_loss, valid_chamfer_loss = [], []
#                 i = 0
#                 for i in range(len(valid_point)//self.batch_size):
#                     valid_s, valid_e = i*self.batch_size, (i+1)*self.batch_size
#                     if len(self.origin_num_points) > 1:
#                         feed_dict = {self.top_pc: valid_point[valid_s:valid_e],
#                                      self.top_meta: valid_meta[valid_s:valid_e],
#                                      # self.low_pc: ,
#                                      # self.low_meta: ,
#                                      self.is_training: False}
#                     else:
#                         feed_dict = {self.top_pc: valid_point[valid_s:valid_e],
#                                      self.top_meta: valid_meta[valid_s:valid_e],
#                                      self.is_training: False}
#                     vel, vcl = self.sess.run([self.emd_loss, self.chamfer_loss], feed_dict)
#                     valid_emd_loss.append(vel)
#                     valid_chamfer_loss.append(vcl)
#                 if len(self.origin_num_points) > 1:
#                         feed_dict = {self.top_pc: valid_point[i*self.batch_size:],
#                                      self.top_meta: valid_meta[i*self.batch_size:],
#                                      # self.low_pc: ,
#                                      # self.low_meta: ,
#                                      self.is_training: False}
#                 else:
#                     feed_dict = {self.pc: valid_point[i*self.batch_size:],
#                                  self.meta: valid_meta[i*self.batch_size:], 
#                                  self.is_training: False}
#                 vel, vcl = self.sess.run([self.emd_loss, self.chamfer_loss], feed_dict)
#                 valid_emd_loss.append(vel)
#                 valid_chamfer_loss.append(vcl)
#                 print ('validation emd loss: {}, validation chamfer loss: {}'.format(np.array(valid_emd_loss).mean(),
#                                                                                          np.array(valid_chamfer_loss).mean()))

#                 def visualize_sample(sample_points, sample_meta, filename=None):
#                     meta_nums = sample_meta['num_points'].values.astype(int)
#                     feed_dict = {self.pc: sample_points, 
#                                  self.meta: meta_nums, 
#                                  self.is_training: False}
#                     sample_generated, sample_emd, sample_chamfer = self.sess.run([self.x_reconstr, 
#                                                                    self.emd_loss, 
#                                                                    self.chamfer_loss], feed_dict)
#                     reconstruction = point_cell.reconstruct_scene(sample_generated, sample_meta)
#                     if filename == None:
#                         sweep = point_cell.test_sweep[point_cell.sample_idx]
#                     elif filename == 'foreground':
#                         sweep = point_cell.test_f_sweep
#                     elif filename == 'background':
#                         sweep = point_cell.test_b_sweep

#                     mean, var, mse = util.sweep_stat(reconstruction, sweep)
#                     print ('sampled sweep emd loss: {}, chamfer loss: {}, MSE: {}'.format(sample_emd,
#                                                                                           sample_chamfer,
#                                                                                           mse))
#                     print ('save reconstructed sweep')
#                     full_filename = 'orig_'+str(epoch) if filename == None else 'orig_{}_{}'.format(filename, str(epoch))
#                     util.visualize_3d_points(sweep, dir=ckpt_path, filename=full_filename)
#                     full_filename = 'recon_'+str(epoch) if filename == None else 'recon_{}_{}'.format(filename, str(epoch))
#                     util.visualize_3d_points(reconstruction, dir=ckpt_path, filename=full_filename)
#                     return reconstruction, sweep
                
#                 # plot
#                 meta_columns = ['index', 'ratio', 'center', 'level', 'num_points'] 
#                 if self.fb_split == True:
#                     sample_points = point_cell.sample_points_f[level_idx].as_matrix(columns=['points'])
#                     sample_points = np.array(list(sample_points.squeeze()))
#                     sample_meta = point_cell.sample_points_f[level_idx][meta_columns]
#                     rf, sf = visualize_sample(sample_points, sample_meta, 'foreground')

#                     sample_points = point_cell.sample_points_b[level_idx].as_matrix(columns=['points'])
#                     sample_points = np.array(list(sample_points.squeeze()))
#                     sample_meta = point_cell.sample_points_b[level_idx][meta_columns]
#                     rb, sb = visualize_sample(sample_points, sample_meta, 'background')
#                     reconstruction = np.concatenate([rf, rb], 0)
#                     sweep = np.concatenate([sf, sb], 0)
#                     util.visualize_3d_points(sweep, dir=ckpt_path, filename='orig_{}'.format(str(epoch)))
#                     util.visualize_3d_points(reconstruction, dir=ckpt_path, filename='recon_{}'.format(str(epoch)))
#                 else:
#                     sample_points = point_cell.sample_points[level_idx].as_matrix(columns=['points'])
#                     sample_points = np.array(list(sample_points.squeeze()))
#                     sample_meta = point_cell.sample_points[level_idx][meta_columns]                  
#                     visualize_sample(sample_points, sample_meta)

#         # check testing accuracy
#         test_emd_loss, test_chamfer_loss = [], []
#         i = 0
#         for i in range(len(test_point)//self.batch_size):
#             test_s, test_e = i*self.batch_size, (i+1)*self.batch_size
#             feed_dict = {self.pc: test_point[test_s:test_e],
#                          self.meta: test_meta[test_s:test_e],
#                          self.is_training: False}
#             vel, vcl = self.sess.run([self.emd_loss, self.chamfer_loss], feed_dict)
#             test_emd_loss.append(vel)
#             test_chamfer_loss.append(vcl)

#         feed_dict = {self.pc: test_point[i*self.batch_size:],
#                      self.meta: test_meta[i*self.batch_size:], 
#                      self.is_training: False}
#         vel, vcl = self.sess.run([self.emd_loss, self.chamfer_loss], feed_dict)
#         test_emd_loss.append(vel)
#         test_chamfer_loss.append(vcl)
#         print ('validation emd loss: {}, validation chamfer loss: {}'.format(np.array(test_emd_loss).mean(),
#                                                                              np.array(test_chamfer_loss).mean()))


                
#     def get_learning_rate(self, batch):
#         learning_rate = tf.train.exponential_decay(
#                             self.learning_rate,  # Base learning rate.
#                             batch * self.batch_size,  # Current index into the dataset.
#                             self.DECAY_STEP,          # Decay step.
#                             self.DECAY_RATE,          # Decay rate.
#                             staircase=True)
#         learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
#         return learning_rate

#     def get_bn_decay(self, batch):
#         bn_momentum = tf.train.exponential_decay(
#                           self.BN_INIT_DECAY,
#                           batch*self.batch_size,
#                           self.BN_DECAY_DECAY_STEP,
#                           self.BN_DECAY_DECAY_RATE,
#                           staircase=True)
#         bn_decay = tf.minimum(self.BN_DECAY_CLIP, 1 - bn_momentum)
#         return bn_decay


#     def preprocess(self, point_cell, level_idx, mode='foreground'):
#         if self.fb_split == True:
#             if mode == 'foreground':
#                 train_point = point_cell.train_cell_f[level_idx].as_matrix(columns=['points'])
#                 train_point = np.array(list(train_point.squeeze()))
#                 valid_point = point_cell.valid_cell_f[level_idx].as_matrix(columns=['points'])
#                 valid_point = np.array(list(valid_point.squeeze()))
#                 test_point = point_cell.test_cell_f[level_idx].as_matrix(columns=['points'])
#                 test_point = np.array(list(test_point.squeeze()))

#                 train_point_l = point_cell.train_cell_f[level_idx].as_matrix(columns=['children'])
#                 train_point_l = np.array(list(train_point_l.squeeze()))
#                 valid_point_l = point_cell.valid_cell_f[level_idx].as_matrix(columns=['children'])
#                 valid_point_l = np.array(list(valid_point_l.squeeze()))
#                 test_point_l = point_cell.test_cell_f[level_idx].as_matrix(columns=['children'])
#                 test_point_l = np.array(list(test_point_l.squeeze()))

#                 train_meta = point_cell.train_cell_f[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)
#                 valid_meta = point_cell.valid_cell_f[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)
#                 test_meta = point_cell.test_cell_f[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)

#             elif mode == 'background':
#                 train_point = point_cell.train_cell_b[level_idx].as_matrix(columns=['points'])
#                 train_point = np.array(list(train_point.squeeze()))
#                 valid_point = point_cell.valid_cell_b[level_idx].as_matrix(columns=['points'])
#                 valid_point = np.array(list(valid_point.squeeze()))
#                 test_point = point_cell.test_cell_b[level_idx].as_matrix(columns=['points'])
#                 test_point = np.array(list(test_point.squeeze()))

#                 train_point_l = point_cell.train_cell_f[level_idx].as_matrix(columns=['children'])
#                 train_point_l = np.array(list(train_point_l.squeeze()))
#                 valid_point_l = point_cell.valid_cell_f[level_idx].as_matrix(columns=['children'])
#                 valid_point_l = np.array(list(valid_point_l.squeeze()))
#                 test_point_l = point_cell.test_cell_f[level_idx].as_matrix(columns=['children'])
#                 test_point_l = np.array(list(test_point_l.squeeze()))


#                 train_meta = point_cell.train_cell_b[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)
#                 valid_meta = point_cell.valid_cell_b[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)
#                 test_meta = point_cell.test_cell_b[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)

#             else:
#                 print ("unkonwn mode {}, should be either foreground or background!" % mode)
#                 raise
#         else:
#             train_point = point_cell.train_cell[level_idx].as_matrix(columns=['points'])
#             train_point = np.array(list(train_point.squeeze()))
#             valid_point = point_cell.valid_cell[level_idx].as_matrix(columns=['points'])
#             valid_point = np.array(list(valid_point.squeeze()))
#             test_point = point_cell.test_cell[level_idx].as_matrix(columns=['points'])
#             test_point = np.array(list(test_point.squeeze()))
        
#             train_meta = point_cell.train_cell[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)
#             valid_meta = point_cell.valid_cell[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)
#             test_meta = point_cell.test_cell[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)

#         return train_point, valid_point, test_point, train_meta, valid_meta, test_meta

#     def predict(self, points, info, ckpt_name, mode='foreground'):

#         meta = np.array([info[i]['num_points'] for i in range(len(info))])
#         self.sess.run(tf.global_variables_initializer())
#         self.saver.restore(self.sess, os.path.join(self.save_path, ckpt_name))
#         feed_dict = {self.pc: np.array(points), self.meta: meta, self.is_training: False}
#         point_reconstr, reconstr_loss = self.sess.run([self.x_reconstr, self.reconstruction_loss], feed_dict)
#         return point_reconstr





class StackAutoEncoder():
    def __init__(self, FLAGS):
        self.batch_size = FLAGS.batch_size        
        self.latent_dim = FLAGS.latent_dim
        self.num_epochs = FLAGS.num_epochs
        self.save_freq = FLAGS.save_freq
        self.save_path = FLAGS.save_path
        self.fb_split = FLAGS.fb_split
        self.level = len(FLAGS.PW)

        self.factor = FLAGS.PL[1] / FLAGS.PL[0] if len(FLAGS.PL) > 1 else 1
        self.learning_rate = FLAGS.learning_rate
        self.loss_type = FLAGS.loss_type
        self.origin_num_points = list(map(int, FLAGS.cell_max_points.split(',')))
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

        self.g = tf.Graph()
        self._build_model(self.g)

    def _encoder(self, net, mask, name_scope):
        with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
            net = tf_util.point_conv(input_image=net, 
                                     is_training=self.is_training, 
                                     n_filters=[64, 64], 
                                     name="encoder1")
            net = tf_util.point_conv(input_image=net, 
                                     is_training=self.is_training, 
                                     n_filters=[64, self.latent_dim],
                                     name="encoder2")

        return mask * net

    def _decoder(self, code, name_scope):
        with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
            if self.decoder_mode == 'pointnet':
                x_reconstr = tf_util.fully_connected_decoder(code, 
                                                             self.origin_num_points[1], 
                                                             self.is_training)
            elif self.decoder_mode == 'pointgrid':
                code = tf.expand_dims(code, -2)
                global_code_tile = tf.tile(code, [1, self.origin_num_points[1], 1])
                local_code_tile = tf_util.fold3d(self.origin_num_points[1], 
                                                 self.map_size,
                                                 tf.shape(code)[0])
                all_code = tf.concat([local_code_tile, global_code_tile], axis=-1)

                net = tf_util.point_conv(all_code, 
                                         self.is_training, 
                                         n_filters=[256, 128, 64], 
                                         activation_function=tf.nn.relu,
                                         name="decoder1")
                net = tf_util.point_conv(net, 
                                         self.is_training, 
                                         n_filters=[3],
                                         activation_function=None,
                                         name="decoder2")
                # second folding
                if self.enable_2ndfold:
                    net = tf.concat([local_code_tile, net], axis=-1)
                    net = tf_util.point_conv(net, 
                                             self.is_training, 
                                             n_filters=[258, 128, 64], 
                                             activation_function=tf.nn.relu,
                                             name="decoder3")
                    net = tf_util.point_conv(net, 
                                             self.is_training, 
                                             n_filters=[3],
                                             activation_function=None,
                                             name="decoder4")
                net = tf.reshape(net, [-1, self.origin_num_points[1], 3])
                net_xy = tf.nn.tanh(net[:,:,0:2])
                net_z = tf.expand_dims(net[:,:,2], -1)
                x_reconstr = tf.concat([net_xy, net_z], -1)
        return self.mask * x_reconstr

    def _fuse(self, code):
        code = tf.expand_dims(tf.expand_dims(code, -2), -2)
        code = tf_util.point_conv(code, 
                         self.is_training, 
                         n_filters=[32, self.latent_dim],
                         activation_function=None,
                         name="fuse")
        return code

    def _ae(self, top_pc, top_mask, low_pc, low_mask):

        # ------- encoder -------
        top_latent = self._encoder(top_pc, top_mask, name_scope='top_encoder')
        top_code = tf.reduce_max(top_latent, axis=-2, keepdims=False)

        if len(self.origin_num_points) > 1:
            # ------- encoder -------
            low_latent = self._encoder(low_pc[:, l], low_mask[:, l], name_scope='low_encoder')
            low_code = tf.reduce_max(low_latent, axis=-2, keepdims=False)

            code = tf.concat([low_code, top_code], axis=-1)

            combined_code = self._fuse(code)
        else:
            combined_code = top_code

        # ------- decoder ------- 
        x_reconstr = self._decoder(combined_code, name_scope='decoder')

        return x_reconstr

    def _build_model(self, g):
        with g.as_default():
            # MLP: all_points [batch, 200, 2] -> MLP -> node_feature [batch, 200, 10]
            self.top_pc = tf.placeholder(tf.float32, [None, self.origin_num_points[0], 3])
            self.top_meta = tf.placeholder(tf.int32, [None])
            top_mask = tf.sequence_mask(self.top_meta, maxlen=self.origin_num_points[0], dtype=tf.float32)
            self.top_mask = tf.expand_dims(top_mask, -1)

            if len(self.origin_num_points) > 1:
                self.low_pc = tf.placeholder(tf.float32, [None, self.factor, self.origin_num_points[1], 3])
                self.low_meta = tf.placeholder(tf.int32, [None, self.factor])
                low_mask = []
                for i in range(self.factor):
                    low_mask.append(tf.sequence_mask(self.low_meta[:,i], maxlen=self.origin_num_points[1], dtype=tf.float32))
                self.low_mask = tf.expand_dims(tf.stack(low_mask, axis=1), -1)

            self.is_training = tf.placeholder(tf.bool, shape=())
            
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)

            self.x_reconstr = self._ae(self.top_pc, self.top_mask, self.low_pc, self.low_mask)

            match = approx_match(self.x_reconstr, self.top_pc)
            self.emd_loss = tf.reduce_mean(match_cost(self.x_reconstr, self.top_pc, match))

            def calculate_chamfer(recon, orig, num_points):
                x_reconstr, pc = tf.expand_dims(recon[:num_points], 0), tf.expand_dims(orig[:num_points], 0)
                cost_p1_p2, _, cost_p2_p1, _ = nn_distance(x_reconstr, pc)
                return tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)

            # self.chamfer_loss = calculate_chamfer(self.x_reconstr, self.pc)
            self.chamfer_loss = tf.reduce_mean(tf.map_fn(lambda x: calculate_chamfer(x[0], x[1], x[2]), 
                                                         (self.x_reconstr, self.top_pc, self.top_meta), dtype=tf.float32))
            #  -------  loss + optimization  ------- 
            if self.loss_type == "emd":
                self.reconstruction_loss = self.emd_loss
            elif self.loss_type == "chamfer":
                self.reconstruction_loss = self.chamfer_loss

            # reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            learning_rate = self.get_learning_rate(batch)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.reconstruction_loss, global_step=batch)
            
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            self.sess = tf.Session(config=config)

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

    def train(self, point_cell, mode='foreground'):
        
        train_point, valid_point, test_point, \
        train_meta, valid_meta, test_meta = self.preprocess(point_cell, self.current_level, mode)

        record_loss = []

        num_batches = train_point.shape[0] // self.batch_size
        for epoch in range(self.num_epochs):

            train_idx = np.arange(0, len(dataset.train_set))
            np.random.shuffle(train_idx)
            current_point = train_point[train_idx]
            current_meta = train_meta[train_idx]

            record_mean, record_var, record_mse = [], [], []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx+1) * self.batch_size

                X = current_point[start_idx:end_idx]
                meta = current_meta[start_idx:end_idx]
                if len(self.origin_num_points) > 1:
                    feed_dict={self.top_pc: X, 
                               # self.low_pc: ,
                               self.top_meta: meta, 
                               # self.low_meta: ,
                               self.is_training: True}
                else:
                    feed_dict={self.top_pc: X, self.top_meta: meta, self.is_training: True}
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
            if epoch % self.save_freq == 0:
                ckpt_path = util.create_dir(osp.join(self.save_path, 'level_%s' % (self.current_level)))
                if self.fb_split == True:
                    ckpt_name = osp.join(ckpt_path, 'model-%s-%s.ckpt' % (mode, str(epoch)))
                else:
                    ckpt_name =  osp.join(ckpt_path, 'model-%s.ckpt' % (str(epoch)))
                
                self.saver.save(self.sess, ckpt_name)

                # evaluation
                valid_emd_loss, valid_chamfer_loss = [], []
                i = 0
                for i in range(len(valid_point)//self.batch_size):
                    valid_s, valid_e = i*self.batch_size, (i+1)*self.batch_size
                    if len(self.origin_num_points) > 1:
                        feed_dict = {self.top_pc: valid_point[valid_s:valid_e],
                                     self.top_meta: valid_meta[valid_s:valid_e],
                                     # self.low_pc: ,
                                     # self.low_meta: ,
                                     self.is_training: False}
                    else:
                        feed_dict = {self.top_pc: valid_point[valid_s:valid_e],
                                     self.top_meta: valid_meta[valid_s:valid_e],
                                     self.is_training: False}
                    vel, vcl = self.sess.run([self.emd_loss, self.chamfer_loss], feed_dict)
                    valid_emd_loss.append(vel)
                    valid_chamfer_loss.append(vcl)
                if len(self.origin_num_points) > 1:
                        feed_dict = {self.top_pc: valid_point[i*self.batch_size:],
                                     self.top_meta: valid_meta[i*self.batch_size:],
                                     # self.low_pc: ,
                                     # self.low_meta: ,
                                     self.is_training: False}
                else:
                    feed_dict = {self.pc: valid_point[i*self.batch_size:],
                                 self.meta: valid_meta[i*self.batch_size:], 
                                 self.is_training: False}
                vel, vcl = self.sess.run([self.emd_loss, self.chamfer_loss], feed_dict)
                valid_emd_loss.append(vel)
                valid_chamfer_loss.append(vcl)
                print ('validation emd loss: {}, validation chamfer loss: {}'.format(np.array(valid_emd_loss).mean(),
                                                                                         np.array(valid_chamfer_loss).mean()))

                def visualize_sample(sample_points, sample_meta, filename=None):
                    meta_nums = sample_meta['num_points'].values.astype(int)
                    feed_dict = {self.pc: sample_points, 
                                 self.meta: meta_nums, 
                                 self.is_training: False}
                    sample_generated, sample_emd, sample_chamfer = self.sess.run([self.x_reconstr, 
                                                                   self.emd_loss, 
                                                                   self.chamfer_loss], feed_dict)
                    reconstruction = point_cell.reconstruct_scene(sample_generated, sample_meta)
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
                meta_columns = ['index', 'ratio', 'center', 'level', 'num_points'] 
                if self.fb_split == True:
                    sample_points = point_cell.sample_points_f[level_idx].as_matrix(columns=['points'])
                    sample_points = np.array(list(sample_points.squeeze()))
                    sample_meta = point_cell.sample_points_f[level_idx][meta_columns]
                    rf, sf = visualize_sample(sample_points, sample_meta, 'foreground')

                    sample_points = point_cell.sample_points_b[level_idx].as_matrix(columns=['points'])
                    sample_points = np.array(list(sample_points.squeeze()))
                    sample_meta = point_cell.sample_points_b[level_idx][meta_columns]
                    rb, sb = visualize_sample(sample_points, sample_meta, 'background')
                    reconstruction = np.concatenate([rf, rb], 0)
                    sweep = np.concatenate([sf, sb], 0)
                    util.visualize_3d_points(sweep, dir=ckpt_path, filename='orig_{}'.format(str(epoch)))
                    util.visualize_3d_points(reconstruction, dir=ckpt_path, filename='recon_{}'.format(str(epoch)))
                else:
                    sample_points = point_cell.sample_points[level_idx].as_matrix(columns=['points'])
                    sample_points = np.array(list(sample_points.squeeze()))
                    sample_meta = point_cell.sample_points[level_idx][meta_columns]                  
                    visualize_sample(sample_points, sample_meta)

        # check testing accuracy
        test_emd_loss, test_chamfer_loss = [], []
        i = 0
        for i in range(len(test_point)//self.batch_size):
            test_s, test_e = i*self.batch_size, (i+1)*self.batch_size
            feed_dict = {self.pc: test_point[test_s:test_e],
                         self.meta: test_meta[test_s:test_e],
                         self.is_training: False}
            vel, vcl = self.sess.run([self.emd_loss, self.chamfer_loss], feed_dict)
            test_emd_loss.append(vel)
            test_chamfer_loss.append(vcl)

        feed_dict = {self.pc: test_point[i*self.batch_size:],
                     self.meta: test_meta[i*self.batch_size:], 
                     self.is_training: False}
        vel, vcl = self.sess.run([self.emd_loss, self.chamfer_loss], feed_dict)
        test_emd_loss.append(vel)
        test_chamfer_loss.append(vcl)
        print ('validation emd loss: {}, validation chamfer loss: {}'.format(np.array(test_emd_loss).mean(),
                                                                             np.array(test_chamfer_loss).mean()))


                
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


    def preprocess(self, point_cell, level_idx, mode='foreground'):
        if self.fb_split == True:
            if mode == 'foreground':
                train_point = point_cell.train_cell_f[level_idx].as_matrix(columns=['points'])
                train_point = np.array(list(train_point.squeeze()))
                valid_point = point_cell.valid_cell_f[level_idx].as_matrix(columns=['points'])
                valid_point = np.array(list(valid_point.squeeze()))
                test_point = point_cell.test_cell_f[level_idx].as_matrix(columns=['points'])
                test_point = np.array(list(test_point.squeeze()))

                train_point_l = point_cell.train_cell_f[level_idx].as_matrix(columns=['children'])
                train_point_l = np.array(list(train_point_l.squeeze()))
                valid_point_l = point_cell.valid_cell_f[level_idx].as_matrix(columns=['children'])
                valid_point_l = np.array(list(valid_point_l.squeeze()))
                test_point_l = point_cell.test_cell_f[level_idx].as_matrix(columns=['children'])
                test_point_l = np.array(list(test_point_l.squeeze()))

                train_meta = point_cell.train_cell_f[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)
                valid_meta = point_cell.valid_cell_f[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)
                test_meta = point_cell.test_cell_f[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)

            elif mode == 'background':
                train_point = point_cell.train_cell_b[level_idx].as_matrix(columns=['points'])
                train_point = np.array(list(train_point.squeeze()))
                valid_point = point_cell.valid_cell_b[level_idx].as_matrix(columns=['points'])
                valid_point = np.array(list(valid_point.squeeze()))
                test_point = point_cell.test_cell_b[level_idx].as_matrix(columns=['points'])
                test_point = np.array(list(test_point.squeeze()))

                train_point_l = point_cell.train_cell_f[level_idx].as_matrix(columns=['children'])
                train_point_l = np.array(list(train_point_l.squeeze()))
                valid_point_l = point_cell.valid_cell_f[level_idx].as_matrix(columns=['children'])
                valid_point_l = np.array(list(valid_point_l.squeeze()))
                test_point_l = point_cell.test_cell_f[level_idx].as_matrix(columns=['children'])
                test_point_l = np.array(list(test_point_l.squeeze()))


                train_meta = point_cell.train_cell_b[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)
                valid_meta = point_cell.valid_cell_b[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)
                test_meta = point_cell.test_cell_b[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)

            else:
                print ("unkonwn mode {}, should be either foreground or background!" % mode)
                raise
        else:
            train_point = point_cell.train_cell[level_idx].as_matrix(columns=['points'])
            train_point = np.array(list(train_point.squeeze()))
            valid_point = point_cell.valid_cell[level_idx].as_matrix(columns=['points'])
            valid_point = np.array(list(valid_point.squeeze()))
            test_point = point_cell.test_cell[level_idx].as_matrix(columns=['points'])
            test_point = np.array(list(test_point.squeeze()))
        
            train_meta = point_cell.train_cell[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)
            valid_meta = point_cell.valid_cell[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)
            test_meta = point_cell.test_cell[level_idx].as_matrix(columns=['num_points']).squeeze().astype(int)

        return train_point, valid_point, test_point, train_meta, valid_meta, test_meta

    def predict(self, points, info, ckpt_name, mode='foreground'):

        meta = np.array([info[i]['num_points'] for i in range(len(info))])
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, os.path.join(self.save_path, ckpt_name))
        feed_dict = {self.pc: np.array(points), self.meta: meta, self.is_training: False}
        point_reconstr, reconstr_loss = self.sess.run([self.x_reconstr, self.reconstruction_loss], feed_dict)
        return point_reconstr
