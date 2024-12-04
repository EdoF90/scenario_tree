# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from abc import abstractmethod
import matplotlib.pyplot as plt

class StochModel():
    @abstractmethod
    def __init__(self, sim_setting):
        self.initial_share_price = sim_setting['initial_share_prices']
        self.tickers = sim_setting['tickers']
        self.n_shares = len(self.tickers)
        self.risk_free_return = sim_setting['risk_free_return']

    @abstractmethod
    def simulate_one_time_step(self, parent_node, n_children):
        pass
    
    @abstractmethod
    def simulate_all_horizon(self, time_horizon):
        pass
