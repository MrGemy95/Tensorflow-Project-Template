from base.base_model import BaseModel
import tensorflow as tf


class CifarModel(BaseModel):
    def __init__(self, data_loader, config):
        super(CifarModel, self).__init__(config)
        # Get the data_loader to make the joint of the inputs in the graph
        self.data_loader = data_loader

        # define some important variables
        self.x = None
        self.y = None
        self.is_training = None
        self.out_argmax = None
        self.loss = None
        self.acc = None
        self.optimizer = None
        self.train_step = None

        self.build_model()
        self.init_saver()

    def build_model(self):
        """

        :return:
        """
        
        """
        Helper Variables
        """
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.global_step_inc = self.global_step_tensor.assign(self.global_step_tensor + 1)
        self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(self.global_epoch_tensor + 1)
        
        """
        Inputs to the network
        """
        with tf.variable_scope('inputs'):
            self.x, self.y = self.data_loader.get_input()
            self.is_training = tf.placeholder(tf.bool, name='Training_flag')
        tf.add_to_collection('inputs', self.x)
        tf.add_to_collection('inputs', self.y)
        tf.add_to_collection('inputs', self.is_training)

        """
        Network Architecture
        """
        with tf.variable_scope('network'):
            conv1 = CifarModel.conv_bn_relu('conv1_block', self.x, 16, (3, 3), self.is_training)
            conv2 = CifarModel.conv_bn_relu('conv2_block', conv1, 32, (3, 3), self.is_training)

            with tf.variable_scope('max_pool1'):
                max_pool1 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), name='max_pool')

            conv3 = CifarModel.conv_bn_relu('conv3_block', max_pool1, 32, (3, 3), self.is_training)
            conv4 = CifarModel.conv_bn_relu('conv4_block', conv3, 32, (3, 3), self.is_training)

            with tf.variable_scope('max_pool2'):
                max_pool2 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2), name='max_pool')

            with tf.variable_scope('flatten'):
                flattened = tf.layers.flatten(max_pool2, name='flatten')

            dense1 = CifarModel.dense_bn_relu_dropout('dense1', flattened, 512, 0.5, self.is_training)
            dense2 = CifarModel.dense_bn_relu_dropout('dense2', dense1, 256, 0.3, self.is_training)

            with tf.variable_scope('out'):
                self.out = tf.layers.dense(dense2, self.config.num_classes,
                                           kernel_initializer=tf.initializers.truncated_normal, name='out')
                tf.add_to_collection('out', self.out)

        """
        Some operators for the training process
        """
        with tf.variable_scope('out_argmax'):
            self.out_argmax = tf.argmax(self.out, axis=-1, output_type=tf.int64, name='out_argmax')

        with tf.variable_scope('loss-acc'):
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.out)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(self.y, self.out_argmax), tf.float32))

        with tf.variable_scope('train_step'):
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.loss)
        tf.add_to_collection('train', self.acc)

    def init_saver(self):
        """
        initialize the tensorflow saver that will be used in saving the checkpoints.
        :return:
        """
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)

    @staticmethod
    def conv_bn_relu(name, x, out_filters, kernel_size, training_flag):
        with tf.variable_scope(name):
            out = tf.layers.conv2d(x, out_filters, kernel_size, padding='SAME',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv')
            out = tf.layers.batch_normalization(out, training=training_flag, name='bn')
            out = tf.nn.relu(out)
            return out

    @staticmethod
    def dense_bn_relu_dropout(name, x, num_neurons, dropout_rate, training_flag):
        with tf.variable_scope(name):
            out = tf.layers.dense(x, num_neurons, kernel_initializer=tf.initializers.truncated_normal, name='dense')
            out = tf.layers.batch_normalization(out, training=training_flag, name='bn')
            out = tf.nn.relu(out)
            out = tf.layers.dropout(out, dropout_rate, training=training_flag, name='dropout')
            return out
