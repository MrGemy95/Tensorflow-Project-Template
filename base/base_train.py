import tensorflow as tf


class BaseTrain:
    def __init__(self, sess, model, config, logger, data_loader=None):
        """
        Constructing the trainer
        :param sess: TF.Session() instance
        :param model: The model instance
        :param config: config namespace which will contain all the configurations you have specified in the json
        :param logger: logger class which will summarize and write the values to the tensorboard
        :param data_loader: The data loader if specified. (You will find Dataset API example)
        """
        # Assign all class attributes
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        if data_loader is not None:
            self.data_loader = data_loader

        # Initialize all variables of the graph
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self):
        """
        This is the main loop of training
        Looping on the epochs
        :return:
        """
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self, epoch=None):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary

        :param epoch: take the number of epoch if you are interested
        :return:
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step

        - run the tensorflow session
        :return: any metrics you need to summarize
        """
        raise NotImplementedError
