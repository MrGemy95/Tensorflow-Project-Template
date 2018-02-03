import tensorflow as tf
import os


def create_saver(function_to_decorate):

    def wrapper(self, *args, **kw):
        output = function_to_decorate(self, *args, **kw)
        self.init_saver()
        print("post_code")
        return output

    return wrapper


class BaseModel:
    def __init__(self, config, child):
        self.config = config
        # init the global step, global time step, the current epoch and the summaries
        self.init_global_step()
        self.init_cur_epoch()

        self.summaries = None
        self.saver = None

        child.build_model = create_saver(child.build_model)


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
        # DON'T forget to add the global step tensor to the train step
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        print("Saver initiated")

    @classmethod
    def cls_create_saver(cls, func):
        pass
    def build_model(self):
        raise NotImplementedError
