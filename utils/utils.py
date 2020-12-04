import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-t', '--train',
        default=False,
        action='store_true',
        help='Specify the process training or validation, default value is False for validation')
    args = argparser.parse_args()
    return args
