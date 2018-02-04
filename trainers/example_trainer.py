from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np

class ExampleTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(ExampleTrainer, self).__init__(sess, model, data, config,logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses=[]
        accs=[]
        for it in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss=np.mean(losses)
        acc=np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {}
        summaries_dict['loss'] = loss
        summaries_dict['acc'] = acc
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc
