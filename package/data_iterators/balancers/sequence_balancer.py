"""
NeurodataLab LLC 01.02.2019
Created by Andrey Belyaev
"""
import logging
import numpy as np

from .base_balancer import BaseBalancer
from ... import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL

logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


class SequenceBalancer(BaseBalancer):
    """
    Balances sequences (pairs, triplets, ...) from given indexes

    Example (for pairs):
        data = [1, 2, 3, 4]
        b.next()  # 1, 3
        b.next()  # 2, 4
    """
    def __init__(self, balancer, sequence_len):
        assert sequence_len > 0

        self.balancer = balancer
        self.sequence_len = sequence_len

        super(SequenceBalancer, self).__init__(None)

    def _reset(self):
        self.balancer.reset()

    def next(self):
        return [self.balancer.next() for _ in range(self.sequence_len)]
