#used for global access
global net_config

def __init(config):
    global net_config
    net_config = config

def set_cfg(config):
    global net_config
    net_config = config

def get_cfg():
    global net_config
    try:
        return net_config
    except KeyError:
        return None