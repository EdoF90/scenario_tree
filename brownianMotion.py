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
            self.start_date,
            self.end_date
        ) # Moments estimated from historical data are used as parameteres of the GBM

    def estimate_from_data(self, start, end):
        hist_prices = yf.download(
            self.tickers,
            start = start,
            end = end
        )['Close']
        monthly_prices = hist_prices.resample('ME').first()
        log_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()
        self.mu = log_returns.mean().values
        self.sigma = log_returns.std().values
        self.corr = log_returns.corr().values

    def simulate_one_time_step(self, n_children, parent_node):
        # Inizialization
        arb = True
        counter = 0
        # Main loop: keeps sampling until an arbitrage-free solution is found (or until the maximum number of iterations is reached)
        while (arb == True) and (counter<10000):
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
            returns = np.array(c).reshape(-1,1) * self.dt + Y

            # Transform returns to prices
            prices = np.zeros((self.n_shares, n_children))
            for i in range(self.n_shares):
                for s in range(n_children):
                    prices[i,s] = parent_node[i] * np.exp(returns[i,s]) 

            # Check the presence of arbitrage opportunities
            arb = check_arbitrage_prices(prices, parent_node)
            if (arb == False):
                logging.info(f"No arbitrage solution found after {counter} iteration(s)")
        if counter >= 10000:
            raise RuntimeError(f"No arbitrage solution NOT found after {counter} iteration(s)")
        else:
            probs = 1/n_children * np.ones(n_children)
            return probs, prices # return probabilities and price to add the generated nodes to the tree
    
