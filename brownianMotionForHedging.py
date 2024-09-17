# -*- coding: utf-8 -*-
from .stochModel import StochModel
import yfinance as yf
from scipy import stats
import numpy as np
from .checkarbitrage import check_arbitrage_prices
import logging

class BrownianMotionForHedging(StochModel):
    ''' BrownianMotionForHedging: stochastic model used to simulate stock price dynamics 
        uder the Geometric Brownian Motion model.
        simulate_one_time_step: for each parent node in scenario tree, it generates children 
        nodes by computing new asset values and the probabilities of each new node.
        Stock prices following Geometric Brownian Motion are generated until a no arbitrage 
        setting is found. If the market is arbitrage free, option prices (using Black and Scholes formula)
        and cash new values are computed.
    '''

    def __init__(self, 
                 sim_setting, 
                 option_list, 
                 dt, mu, 
                 sigma, rho, 
                 rnd_state): 
        
        super().__init__(sim_setting)
        self.dt = dt 
        self.n_options = len(option_list)
        self.option_list = option_list
        self.mu = mu
        self.sigma = sigma
        self.corr = rho 
        self.rnd_state = rnd_state


    def simulate_one_time_step(self, n_children, parent_node, remaining_times): 
        # find the values (prices) of each asset for each node 
        # in the new step with the same parent node
        parent_stock_prices= parent_node[1:self.n_shares+1] 
        parent_cash_price = parent_node[0]

        arb = True
        counter = 0
        # Simulate stock prices until no-arbitarge is found:
        while (arb == True) and (counter<100):
            counter += 1
            # In simulations settings (Geometric Brownian Motion): 
            # S(t+dt) = S(t) * exp((mu - 1/2*sigma**2) * dt + sigma * sqrt(dt) * Z)
            # where Z is a standard normal distribution
            if self.n_shares > 1:
                B = self.rnd_state.multivariate_normal(
                    mean = np.zeros(self.n_shares),
                    cov  = self.corr,
                    size = n_children, # verify the correctness of the size
                    ).T    
            else: 
                B = self.rnd_state.normal(loc = 0, scale = 1, size=n_children)
        
            Y = np.array(self.sigma).reshape(-1,1) * np.sqrt(self.dt) * B
            c = self.mu - 0.5 * self.sigma**2
            Increment = np.array(c).reshape(-1,1) * self.dt + Y

            stock_prices = np.zeros((self.n_shares, n_children))
            for i in range(self.n_shares):
                for s in range(n_children):
                    stock_prices[i,s] = parent_stock_prices[i] * np.exp(Increment[i,s]) 

            arb = check_arbitrage_prices(stock_prices, parent_stock_prices)
            if (arb == False):
                logging.info(f"No arbitrage solution found after {counter} iteration(s)")
            
        if counter >= 100:
            raise RuntimeError(f"No arbitrage solution NOT found after {counter} iteration(s)")
        else:
            probs = 1/n_children * np.ones(n_children) #TODO: uniform probabilities ? 
        
        # Options values
        option_prices = np.zeros((self.n_options, n_children))
        for j in range(self.n_shares):
            S0 = stock_prices[j,:]
            time_to_maturity = remaining_times * self.dt 
            # hedging options are assumed to be of European type
            option_prices[j,:] = self.option_list[j].BlackScholesPrice(S0, time_to_maturity)

        # Cash value
        cash_price = parent_cash_price * np.exp(self.option_list[0].risk_free_rate*self.dt) * np.ones(shape=n_children)

        prices = np.vstack((cash_price, stock_prices, option_prices))
        
        return probs, prices
