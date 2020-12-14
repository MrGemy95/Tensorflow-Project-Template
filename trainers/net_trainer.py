from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf

class NetTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(NetTrainer, self).__init__(sess, model, data, config, logger)

        self.train_step_function = {
            "1":self.train_step_v1,
            "2":self.train_step_v2
        }

        self.valid_step_function = {
            "1":self.valid_step_v1,
            "2":self.valid_step_v2
        }

    def train_epoch(self):
        #init the dataset for each epoch
        #self.data.init_train_iter(self.sess)
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.numpy()
        summaries_dict = {
            'loss_epoch': loss,
            'acc_epoch': acc,
        }
        self.logger.summarize(cur_it, summarizer = "train", summaries_dict=summaries_dict)
        self.model.save(self.sess)
        print("Train-loss:{}, acc:{}".format(loss, acc))

        #init the dataset for each epoch
        #self.data.init_val_iter(self.sess)
        loop = tqdm(range(int(self.data.val_img_cnt * self.config.ratio_valid)))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.valid_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.numpy()
        summaries_dict = {
            'loss_epoch': loss,
            'acc_epoch': acc,
        }
        self.logger.summarize(cur_it, summarizer = "test", summaries_dict=summaries_dict)

        print("Valid-loss:{}, acc:{}".format(loss, acc))

    def train_step_v1(self):
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

    def valid_step_v1(self):
        feed_dict = {self.model.is_training: False}
        loss, acc = self.sess.run([self.model.loss, self.model.accuracy], feed_dict=feed_dict)

        return loss, acc

    def train_step_v2(self):
        def _train_step(model, images, labels):
            with tf.GradientTape() as tape:
                logits = model.net(images, training=True)
                loss = model.loss(logits, labels)
            # compute gradient
            grads = tape.gradient(loss, model.net.trainable_variables)
            # update to weights
            model.optimizer.apply_gradients(zip(grads, model.net.trainable_variables))

            acc = model.accuracy(logits, labels)
            # loss and accuracy is scalar tensor
            return logits, loss, acc

        batch_x, batch_y = self.data.get_next(data = "train")
        _, loss, acc = _train_step(self.model, batch_x, batch_y)

        step = self.model.update_global_step()
        return loss, acc

    def valid_step_v2(self):
        def _test_step(model, images, labels):
            logits = model.net(images, training=False)
            loss = model.loss(logits, labels)
            acc = model.accuracy(logits, labels)
            # loss and accuracy is scalar tensor
            return logits, loss, acc

        batch_x, batch_y = self.data.get_next(data = "val")
        _, loss, acc = _test_step(self.model, batch_x, batch_y)

        #step = self.model.update_global_step()
        return loss, acc

    def train_step(self):
        return self.train_step_function[str(self.config.tf_version[0])]()

    def valid_step(self):
        return self.valid_step_function[str(self.config.tf_version[0])]()
