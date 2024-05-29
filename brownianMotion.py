# -*- coding: utf-8 -*-
from .stochModel import StochModel
import yfinance as yf
from scipy import stats
import numpy as np
from numpy import random
#To complete
class BrownianMotion(StochModel):

    def __init__(self, sim_setting):
        super().__init__(sim_setting)
        self.mu = sim_setting["mu"]
        self.sigma = sim_setting["sigma"]
        self.dt = sim_setting["dt"]

    def ExpectedMomentsEstimate(self, start, end):
        data = yf.download(
            self.tickers,
            start= start,#'2023-05-01',
            end= end#'2024-05-01'
        )['Adj Close']
        returns = data.pct_change().dropna()
        self.exp_mean = returns.mean().values
        self.exp_std = returns.std().values
        self.exp_skew = returns.apply(
            lambda x: stats.skew(x, bias=False)
        ).values
        self.exp_kur = returns.apply(
            lambda x: stats.kurtosis(x, bias=False, fisher=False)
        ).values
        self.exp_cor = returns.corr().values

    def simulate_one_time_step(self, n_children, parent_node):
        if self.n_shares > 1:
            B = random.multivariate_normal(
                mean = np.zeros(self.n_underlyings),
                cov  = self.rho,
                size = n_children,
                ).T    
        else: 
            B = random.normal(loc = 0, scale = 1, size=n_children)
        
        Y = np.array(self.sigma).reshape(-1,1) * np.sqrt(self.dt) * B
        c = self.mu - 0.5 * self.sigma**2
        Increment = np.array(c).reshape(-1,1) * self.dt + Y

        #TODO to complete
        prices[1:n_u+1, :] = parent_node[1:n_u+1].reshape(-1,1) * np.exp( Inc )


        return prices

    def generate_states(self):
        '''
        Generate children states by pure MC simulation.
        '''
        size = (self.branching_factor,)  #((self.n_underlyings, self.branching_factor))
        if self.dynamics == 'BS':
            if self.n_shares > 1:
                B = random.multivariate_normal(
                    mean = np.zeros(self.n_underlyings),
                    cov  = self.rho,
                    size = size,  #n_underlyings is automatic
                    ).T
            else: B = self.Obj.normal(loc = 0, scale = 1, size=size )
            Y = np.array(self.sigma).reshape(-1,1) * np.sqrt(self.dt) * B
            # c_rn   = self.r - 0.5 * self.sigma**2
            c_hist = self.mu - 0.5 * self.sigma**2
        if self.dynamics == 'VG':
            G = self.Obj.gamma( shape = self.dt/self.nu, scale = self.nu, size=size ) # scale = 1 / rate
            Y = self.Obj.normal( loc = self.mu*G, scale = self.sigma*np.sqrt(G), size=size )
            # c_rn   = self.r + np.log(1 - self.nu*self.mu - self.nu*self.sigma**2/2) / self.nu        
            c_hist = self.c
        # c = c_rn  #-> if we want a risk-neutral 'c'
        c = c_hist
        Inc = np.array(c).reshape(-1,1) * self.dt + Y
        n_u = self.n_underlyings
        self.next_states[1:n_u+1, :] = self.current_state[1:n_u+1].reshape(-1,1) * np.exp( Inc )
