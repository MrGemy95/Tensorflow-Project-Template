import tensorflow as tf


class BasePredict:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        if self.config.tf_version[0] < 2:
            self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.sess.run(self.init)
        else:
            pass

    def predict(self):
        self.predict_epoch()
        self.model.save()

    def predict_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def predict_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
