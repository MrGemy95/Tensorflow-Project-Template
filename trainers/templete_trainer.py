from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class TempleteTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(TempleteTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop ever the number of iteration in the config and call teh train step
       -add any summaries you want using the sammary
        """
        pass

    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
        pass
