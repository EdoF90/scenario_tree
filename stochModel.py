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
        # self.trans_cost_v = sim_setting['trans_cost_v']
        # self.trans_cost_a = sim_setting['trans_cost_a']
        # self.initial_liquidity = sim_setting['initial_liquidity']

    @abstractmethod
    def simulate_one_time_step(self, parent_node, n_children):
        # parent_node è un nodo di ScenarioTree
        # oppure passare ScenarioTree + nome nodo
        # simulate_one_time_step(self, ScenarioTree, id_parent_node, n_children)
        pass
    
    @abstractmethod
    def simulate_all_horizon(self, time_horizon):
        pass
     
    # def graphical_evaluation(self, id_share, time_horizon):
    #     for i in range(self.n_shares):
    #         forecast_1 = self.simulate_all_horizon(time_horizon)
    #         forecast_2 = self.simulate_all_horizon(time_horizon)
    #         plt.plot(self.real_evo[i,0:time_horizon])
    #         plt.plot(forecast_1[i,0:time_horizon], alpha=0.4)
    #         plt.plot(forecast_2[i,0:time_horizon], alpha=0.4)
    #         plt.title(f"Share {i}")
    #         plt.show()
