import tensorflow as tf


class BaseTrain:
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

        self.train_function = {
            "1":self.train_v1,
            "2":self.train_v2
        }

    def train_v1(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_v2(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.numpy(), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.model.cur_epoch_tensor.assign_add(1)
            self.model.save()

    def train(self):
        self.train_function[str(self.config.tf_version[0])]()

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def valid_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
