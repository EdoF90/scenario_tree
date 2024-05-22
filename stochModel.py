from abc import abstractmethod
import numpy as np
import pandas as pd
from solver import *
import matplotlib.pyplot as plt

class StochModel():
    @abstractmethod
    def __init__(self, sim_setting):
        self.initial_share_price = sim_setting['initial_share_prices']
        self.n_shares = sim_setting['n_shares']
        self.risk_free_return = sim_setting['risk_free_return']
        self.trans_cost_v = sim_setting['trans_cost_v']
        self.trans_cost_a = sim_setting['trans_cost_a']
        self.initial_liquidity = sim_setting['initial_liquidity']
        self.from_file = False

    @abstractmethod
    def simulate_one_time_step(self, n_children):
        pass

    @abstractmethod
    def simulate_all_horizon(self, time_horizon):
        pass
    
    # Old methods that probably are currently unnecessary
    def read_data_from_file(self, file_path):
        df = pd.read_csv(file_path)
        # removing time
        df = df.iloc[: , 1:]
        # each row is the evolution of a share
        self.real_evo = df.to_numpy().T
        # UNDERSAMPLING (ONE OBS PER MONTH)
        self.real_evo = self.real_evo[:, 1::30]
        self.from_file = True
        # UPDATE SETTINGS
        self.n_shares = self.real_evo.shape[0]
        self.initial_share_price = self.real_evo[:, 0]
        # consider that we can buy 10 shares for each company
        self.initial_liquidity = 10 * sum(self.real_evo[:,0])
        self.risk_free_return = min(
            abs(
                (self.real_evo[:,2] - self.real_evo[:,1]) /self.real_evo[:,1]
            )
        ) / 5
        # UPDATE INTERNAL DATA
        self.n_shares = self.real_evo.shape[0]
        self.initial_share_price = self.real_evo[:,0]
     
    def plot_details(self):
        lst_rend = []
        for i in range(self.real_evo.shape[0]):
            lst_rend.append(np.array([(self.real_evo[i, t] - self.real_evo[i, t-1]) / self.real_evo[i, t-1] for t in range(1, self.real_evo.shape[1])] ))
    
    def graphical_evaluation(self, time_horizon):
        for i in range(self.n_shares):
            forecast_1 = self.simulate_all_horizon(time_horizon)
            forecast_2 = self.simulate_all_horizon(time_horizon)
            plt.plot(self.real_evo[i,0:time_horizon])
            plt.plot(forecast_1[i,0:time_horizon], alpha=0.4)
            plt.plot(forecast_2[i,0:time_horizon], alpha=0.4)
            plt.title(f"Share {i}")
            plt.show()
