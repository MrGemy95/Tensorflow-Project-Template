from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class NetTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(NetTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        #init the dataset for each epoch
        self.data.init_train_iter(self.sess)
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss_epoch': loss,
            'acc_epoch': acc,
        }
        self.logger.summarize(cur_it, summarizer = "train", summaries_dict=summaries_dict)
        self.model.save(self.sess)
        print("Train-loss:{}, acc:{}".format(loss, acc))

        #validation
        self.data.init_val_iter(self.sess)
        loop = tqdm(range(int(self.data.val_img_cnt * self.config.ratio_valid)))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.valid_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss_epoch': loss,
            'acc_epoch': acc,
        }
        self.logger.summarize(cur_it, summarizer = "test", summaries_dict=summaries_dict)

        print("Valid-loss:{}, acc:{}".format(loss, acc))

    def train_step(self):
        if self.config.dataset_type != "TF":
            batch_x, batch_y = self.data.next_batch(self.config.batch_size)
            feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        else:
            feed_dict = {self.model.is_training: True}

        _, loss, acc, sum = self.sess.run([self.model.train_step, self.model.loss, self.model.accuracy, self.model.summary],
                                     feed_dict=feed_dict)

        # used to add customer definded summary
        step = self.model.global_step_tensor.eval(self.sess)
        if self.config.num_sum_interval > 0 and step % self.config.num_sum_interval == 0:
            self.logger.train_summary_writer.add_summary(sum, step)

        return loss, acc

    def valid_step(self):
        feed_dict = {self.model.is_training: False}
        loss, acc = self.sess.run([self.model.loss, self.model.accuracy], feed_dict=feed_dict)

        return loss, acc
