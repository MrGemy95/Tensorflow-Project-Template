import tensorflow as tf
import os


class BaseModel:
    def __init__(self, config):
        self.config = config
        self.summaries = None
        # init the global step, global time step, the current epoch and the summaries
        self.init_global_step()
        self.init_cur_epoch()

    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, os.path.join(self.config.checkpoint_dir, self.config.exp_name), self.global_step_tensor)
        print("Model saved")

    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.config.checkpoint_dir, self.config.exp_name))
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    def init_cur_epoch(self):
        """
        Create cur epoch tensor to totally save the process of the training
        :return:
        """
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def init_global_step(self):
        """
        Create a global step variable to be a reference to the number of iterations
        :return:
        """
        #DON'T forget to add the global step tensor to the train step
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def build_model(self):
        raise NotImplementedError

    def init_saver(self):
        # you should implement this function on any child class as  "tf.train.Saver" take snapshot of tf.variables,
        # so you must create the saver after you build the model.
        raise NotImplemented
