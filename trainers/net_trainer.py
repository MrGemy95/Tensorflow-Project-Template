from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf

class NetTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(NetTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        #init the dataset for each epoch
        #self.data.init_train_iter(self.sess)
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            # update global step in training
            self.model.global_step_tensor.assign_add(1)
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

        print("Train-loss:{}, acc:{}".format(loss, acc))

        #init the dataset for each epoch
        #self.data.init_val_iter(self.sess)
        loop = tqdm(range(int(self.data.val_img_cnt * self.config.ratio_valid)))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.valid_step()
            # NONE update global step in validation
            # self.model.global_step_tensor.assign_add(1)
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

    def train_step(self):
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
        return loss, acc

    def valid_step(self):
        def _test_step(model, images, labels):
            logits = model.net(images, training=False)
            loss = model.loss(logits, labels)
            acc = model.accuracy(logits, labels)
            # loss and accuracy is scalar tensor
            return logits, loss, acc

        batch_x, batch_y = self.data.get_next(data = "val")
        _, loss, acc = _test_step(self.model, batch_x, batch_y)
        return loss, acc