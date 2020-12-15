from base.base_predict import BasePredict
from tqdm import tqdm
import numpy as np

class NetPredictor(BasePredict):
    def __init__(self, sess, model, data, config,logger):
        super(NetPredictor, self).__init__(sess, model, data, config, logger)

    def predict_epoch(self):
        #init the dataset for each epoch
        #self.data.init_val_iter(self.sess)
        loop = tqdm(range(self.data.val_img_cnt))
        accs = []
        for _ in loop:
            pred, acc = self.predict_step()
            accs.append(acc)
        acc = np.mean(accs)
        print("Valid acc:{}".format(acc))

    def predict_step(self):
        def _pred_step(model, images, labels):
            logits  = model.net(images, training=False)
            metrics = model.accuracy(logits, labels)
            return logits, metrics

        batch_x, batch_y = self.data.get_next(data = "val")
        logits, metrics = _pred_step(self.model, batch_x, batch_y)
        pred = self.model.pred(logits)
        _    = self.model.pred_proc(logits, batch_x, batch_y)
        return pred, metrics