import tensorflow as tf


class BaseModel:
    def __init__(self, config):
        self.config = config
        if self.config.tf_version[0] < 2:
            # init the global step
            self.init_global_step()
            # init the epoch counter
            self.init_cur_epoch()
        else:
            self.init_vals_v2()

        self.save_function = {
            "1": self.save_v1,
            "2": self.save_v2
        }

        self.load_function = {
            "1": self.load_v1,
            "2": self.load_v2
        }
    # save function that saves the checkpoint in the path defined in the config file
    def save_v1(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load_v1(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # save function that saves the checkpoint in the path defined in the config file
    def save_v2(self, sess = None):
        raise NotImplementedError

    # load latest checkpoint from the experiment path defined in the config file
    def load_v2(self, sess = None):
        raise NotImplementedError

    def save(self, *args):
        self.save_function[str(self.config.tf_version[0])](*args)

    def load(self, *args):
        self.load_function[str(self.config.tf_version[0])](*args)

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    # this is only used in tensorflow 2
    def init_vals_v2(self):
        with tf.name_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.name_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
