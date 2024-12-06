# -*- coding: utf-8 -*-
import logging
import numpy as np
import yfinance as yf
from numpy import random
from .stochModel import StochModel
from .checkarbitrage import check_arbitrage_prices


class BrownianMotion(StochModel):
    ''' 
    Stochastic model used to simulate stock price dynamics 
    under the Geometric Brownian Motion model. Children nodes 
    probabilities are uniform.
    '''

    def __init__(self, sim_setting):
        self.tickers = sim_setting['tickers']
        self.n_shares = len(self.tickers)
        self.start_date = sim_setting["start"]
        self.end_date = sim_setting["end"]
        self.dt = 1
        # Moments estimated from historical data are used as parameteres of the GBM
        self.estimate_from_data(
            self.start_date,
            self.end_date
        ) 


    def estimate_from_data(self, start, end):
        ''' Historical moments estimate.'''

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
        ''' 
        It generates children nodes by computing new asset values and 
        the probabilities of each new node. Stock prices are generated 
        until a no arbitrage setting is found.
        '''
        
        # Inizialization
        arb = True
        counter = 0

        # Main loop: keeps sampling until an arbitrage-free solution is found 
        # (or until the maximum number of iterations is reached)
        while (arb == True) and (counter<10000):
            counter += 1
            
            # In simulations settings (Geometric Brownian Motion): 
            # S(t+dt) = S(t) * exp((mu - 1/2*sigma**2) * dt + sigma * sqrt(dt) * Z)
            # where Z is a standard normal distribution
            if self.n_shares > 1:
                Z = random.multivariate_normal(
                    mean = np.zeros(self.n_shares),
                    cov  = self.corr,
                    size = n_children,
                    ).T    
            else: 
                Z = random.normal(loc = 0, scale = 1, size=n_children)

            # Stochastic term: sigma * sqrt(dt) * Z
            Y = np.array(self.sigma).reshape(-1,1) * np.sqrt(self.dt) * Z

            # Deterministic term: mu - 1/2*sigma**2
            c = self.mu - 0.5 * self.sigma**2

            # Stock prices increment
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
            arb = False
        if counter >= 10000:
            raise RuntimeError(f"No arbitrage solution NOT found after {counter} iteration(s)")
        else:
            probs = 1/n_children * np.ones(n_children)
            return probs, prices # return probabilities and prices to add the generated nodes to the tree
    
