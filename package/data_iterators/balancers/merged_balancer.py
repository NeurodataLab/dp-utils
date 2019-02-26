"""
NeurodataLab LLC 01.02.2019
Created by Andrey Belyaev
"""
import logging

from .base_balancer import BaseBalancer
from ... import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL

logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


class MergedBalancer(BaseBalancer):
    """Iterate through set list of balancer, return either multi index or index from current balancer"""
    def __init__(self, balancers, multi_indexes=False):
        self.balancers = balancers
        self.cur_balancer_num = 0
        self.multi_indexes = multi_indexes

        super(MergedBalancer, self).__init__(None)

    def _reset(self):
        for b in self.balancers:
            b.reset()
        self.cur_balancer_num = 0

    def next(self):
        if not self.multi_indexes:
            cur_balancer = self.balancers[self.cur_balancer_num]
            self.cur_balancer_num = (self.cur_balancer_num + 1) % (len(self.balancers))
            return cur_balancer.next()
        else:
            return [b.next() for b in self.balancers]
