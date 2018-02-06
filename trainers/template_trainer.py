from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class TemplateTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(TemplateTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        pass

    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
        pass
