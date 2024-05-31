# -*- coding: utf-8 -*-
from .stochModel import StochModel
import yfinance as yf
from scipy import stats
import numpy as np
from numpy import random
from .checkarbitrage import check_arbitrage_prices
import logging

class BrownianMotion(StochModel):

    def __init__(self, sim_setting):
        super().__init__(sim_setting)
        self.dt = 1
        self.estimate_from_data(
            sim_setting["start"],
            sim_setting["end"]
        )

    def estimate_from_data(self, start, end):
        hist_prices = yf.download(
            self.tickers,
            start = start,
            end = end
        )['Adj Close']
        log_returns = np.log(hist_prices / hist_prices.shift(1)).dropna()
        self.mu = log_returns.mean().values
        self.sigma = log_returns.std().values
        self.corr = log_returns.corr().values

    def simulate_one_time_step(self, n_children, parent_node):
        arb = True
        counter = 0
        while (arb == True) and (counter<100):
            counter += 1
            if self.n_shares > 1:
                B = random.multivariate_normal(
                    mean = np.zeros(self.n_shares),
                    cov  = self.corr,
                    size = n_children,
                    ).T    
            else: 
                B = random.normal(loc = 0, scale = 1, size=n_children)
        
            Y = np.array(self.sigma).reshape(-1,1) * np.sqrt(self.dt) * B
            c = self.mu - 0.5 * self.sigma**2
            Increment = np.array(c).reshape(-1,1) * self.dt + Y

            prices = np.zeros((self.n_shares, n_children))
            for i in range(self.n_shares):
                for s in range(n_children):
                    prices[i,s] = parent_node[i] * np.exp(Increment[i,s]) 

            arb = check_arbitrage_prices(prices, parent_node)
            if (arb == False):
                logging.info(f"No arbitrage solution found after {counter} iteration(s)")
            
        if counter >= 100:
            raise RuntimeError(f"No arbitrage solution NOT found after {counter} iteration(s)")
        else:
            probs = 1/n_children * np.ones(n_children)
            return probs, prices
    '''
    def generate_states(self):
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
        '''
