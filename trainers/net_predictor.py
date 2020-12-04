from base.base_predict import BasePredict
from tqdm import tqdm
import numpy as np

class NetPredictor(BasePredict):
    def __init__(self, sess, model, data, config,logger):
        super(NetPredictor, self).__init__(sess, model, data, config, logger)

    def predict_epoch(self):
        #init the dataset for each epoch
        self.data.init_val_iter(self.sess)
        #one element per epoch
        loop = tqdm(range(self.data.val_img_cnt))
        for _ in loop:
            pred, metrics = self.predict_step()

        print("final metrics:{}".format(metrics))

    def predict_step(self):
        feed_dict = {self.model.is_training: False}
        x_raw, y_raw, pred, metrics, _ = self.sess.run([self.model.x, self.model.y, self.model.pred, self.model.metrics, self.model.metrics_update],
                      feed_dict=feed_dict)

        self.model.post_proc(x_raw, y_raw, pred)
        return pred, metrics
