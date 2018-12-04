import logging
import mxnet as mx
import numpy as np
import os.path as osp

from .. import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


def make_manual_changer(fname, base_lr):
    def change_lr(*args, **kwargs):
        if osp.isfile(fname):
            with open(fname) as f:
                lr = float(f.read())
            logger.info('Setting lr: {}'.format(lr))
        else:
            lr = base_lr
            logger.info('No lr file provided in {}, setting lr: {}'.format(fname, base_lr))
            with open(fname, 'w') as f:
                f.write(str(lr))
        return lr
    return change_lr


def make_nochange_changer(*args, **kwargs):
    def change_lr(lr, losses):
        return lr
    return change_lr


def make_plateau_changer(delta=5e-2, accum_num_batches=1000, min_lr=1e-5, factor=0.1):
    losses_storage = []
    last_means = []
    num_iter = 0

    def change_lr(lr, losses):
        global last_means
        global losses_storage
        global num_iter

        if isinstance(losses[0], mx.nd.NDArray):
            losses = [loss.asnumpy().mean() for loss in losses]
        elif isinstance(losses[0], np.ndarray):
            losses = [loss.mean() for loss in losses]
        # losses are now list of floats

        if len(losses_storage) == 0:
            losses_storage.append([[] for _ in losses])

        for num, loss in enumerate(losses):
            losses_storage[num].append(loss)

        if num_iter % accum_num_batches == 0 and num_iter != 0:
            if len(last_means) == 0:
                last_means = [np.array(loss).mean() for loss in losses_storage]
                logger.info('No change in lr in first run of changer')
                losses_storage = []
                return lr
            else:
                changes = [abs(1 - i / j) for i, j in zip(losses, last_means)]
                change_per_loss = [ch < delta for ch in changes]
                if reduce(lambda x, y: x and y, change_per_loss):
                    logger.info('Changing lr to {} or setting lr to min_lr'.format(lr / factor))
                    last_means = [np.array(loss).mean() for loss in losses_storage]
                    losses_storage = []
                    return lr / factor if lr / factor > min_lr else min_lr
                else:
                    logger.info('No change in lr due to loss relations: {}'.format(changes))
                    last_means = [np.array(loss).mean() for loss in losses_storage]
                    losses_storage = []
                    return lr

    return change_lr
