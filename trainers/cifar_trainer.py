from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np

import tensorflow as tf

from utils.metrics import AverageMeter
from utils.logger import DefinedSummarizer


class CifarTrainer(BaseTrain):
    def __init__(self, sess, model, config, logger, data_loader):
        """
        Constructing the Cifar trainer based on the Base Train..
        Here is the pipeline of constructing
        - Assign sess, model, config, logger, data_loader(if_specified)
        - Initialize all variables
        - Load the latest checkpoint
        - Create the summarizer
        - Get the nodes we will need to run it from the graph
        :param sess:
        :param model:
        :param config:
        :param logger:
        :param data_loader:
        """
        super(CifarTrainer, self).__init__(sess, model, config, logger, data_loader)

        # load the model from the latest checkpoint
        self.model.load(self.sess)

        # Summarizer
        self.summarizer = logger

        self.x, self.y, self.is_training = tf.get_collection('inputs')
        self.train_op, self.loss_node, self.acc_node = tf.get_collection('train')
    
    def train(self):
        """
        This is the main loop of training
        Looping on the epochs
        :return:
        """
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch(cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            self.test(cur_epoch)

    def train_epoch(self, epoch=None):
        """
        Train one epoch
        :param epoch: cur epoch number
        :return:
        """
        # initialize dataset
        self.data_loader.initialize(self.sess, is_train=True)

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_train), total=self.data_loader.num_iterations_train,
                  desc="epoch-{}-".format(epoch))

        loss_per_epoch = AverageMeter()
        acc_per_epoch = AverageMeter()

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            loss, acc = self.train_step()
            # update metrics returned from train_step func
            loss_per_epoch.update(loss)
            acc_per_epoch.update(acc)

        self.sess.run(self.model.global_epoch_inc)

        # summarize
        summaries_dict = {'train/loss_per_epoch': loss_per_epoch.val,
                          'train/acc_per_epoch': acc_per_epoch.val}
        self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)

        self.model.save(self.sess)
        
        print("""
Epoch-{}  loss:{:.4f} -- acc:{:.4f}
        """.format(epoch, loss_per_epoch.val, acc_per_epoch.val))

        tt.close()

    def train_step(self):
        """
        Run the session of train_step in tensorflow
        also get the loss & acc of that minibatch.
        :return: (loss, acc) tuple of some metrics to be used in summaries
        """
        _, loss, acc = self.sess.run([self.train_op, self.loss_node, self.acc_node],
                                     feed_dict={self.is_training: True})
        return loss, acc
    
    def test(self, epoch):
        # initialize dataset
        self.data_loader.initialize(self.sess, is_train=False)

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_test), total=self.data_loader.num_iterations_test,
                  desc="Val-{}-".format(epoch))

        loss_per_epoch = AverageMeter()
        acc_per_epoch = AverageMeter()

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            loss, acc = self.sess.run([self.loss_node, self.acc_node],
                                     feed_dict={self.is_training: False})
            # update metrics returned from train_step func
            loss_per_epoch.update(loss)
            acc_per_epoch.update(acc)

        # summarize
        summaries_dict = {'test/loss_per_epoch': loss_per_epoch.val,
                          'test/acc_per_epoch': acc_per_epoch.val}
        self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)
        
        print("""
Val-{}  loss:{:.4f} -- acc:{:.4f}
        """.format(epoch, loss_per_epoch.val, acc_per_epoch.val))

        tt.close()