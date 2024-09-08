# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from abc import abstractmethod
import matplotlib.pyplot as plt

class StochModel():
    @abstractmethod
    def __init__(self, sim_setting):
        self.tickers = sim_setting['tickers']
        self.n_shares = len(self.tickers)
        self.start_date = sim_setting["start"]
        self.end_date = sim_setting["end"]

    @abstractmethod
    def simulate_one_time_step(self, parent_node, n_children):
        # parent_node is the node that the generate the two-stage subtree that we are going to build and add to the general scenario tree
        pass
    
    '''
    NOT USED METHOD
    @abstractmethod
    def simulate_all_horizon(self, time_horizon):
        pass
    '''