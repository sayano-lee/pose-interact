import logging
# logging.disable(logging.WARNING)
import os
import sys


def make_logger(name, save_dir, save_filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, save_filename+".txt"), mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger