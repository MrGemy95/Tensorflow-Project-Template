import json
from bunch import Bunch
import os
import tensorflow as tf
import utils.config_val as g_config



def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(json_file, is_train=True):
    config, _ = get_config_from_json(json_file)
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, "..")
    config.root_dir         = path + "/"
    config.summary_dir      = os.path.join(path, "experiments", config.exp_name, "summary/")
    config.checkpoint_dir   = os.path.join(path, "experiments", config.exp_name, "checkpoint/")
    config.saved_model_dir  = os.path.join(path, "experiments", config.exp_name, "saved_model/")
    config.update(is_train=is_train)
    config.update(tf_version=[int(i) for i in tf.__version__.split(".")])
    g_config.__init(config)
    return config
