# -*- coding: utf-8 -*-
from abc import abstractmethod

class StochModel():
    @abstractmethod
    def __init__(self, sim_setting):
        pass


    @abstractmethod
    def simulate_one_time_step(self, parent_node, n_children): 
        # parent_node is the node that the generate the two-stage subtree that
        # we are going to build and add to the general scenario tree
        pass
