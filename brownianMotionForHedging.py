# -*- coding: utf-8 -*-
import logging
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy import optimize
from assets import MultiStock
from .stochModel import StochModel
from .checkarbitrage import check_arbitrage_prices, make_arbitrage_free_states
from .calculatemoments import mean, second_moment, correlation

class BrownianMotionForHedging(StochModel):
    ''' 
    Stochastic model used to simulate stock price dynamics 
    under the Geometric Brownian Motion model. The moment matching 
    optimization problem used to find children nodes probabilities 
    is solved with Sequential Least Squares Quadratic Programming. 
    The matched properties are: first and second moment, correlation.
    Options are then priced via Black&Scholes formula.
    '''

    def __init__(self, 
                 tickers, 
                 option_list, 
                 dt, r, mu, 
                 sigma, rho, 
                 skew, kur, 
                 rnd_state): 
        
        self.n_shares = len(tickers)
        self.dt = dt 
        self.n_options = len(option_list)
        self.option_list = option_list
        self.risk_free_rate = r
        self.mu = mu
        self.sigma = sigma
        self.corr = rho 
        self.skew = skew
        self.kur = kur
        self.rnd_state = rnd_state


    def simulate_one_time_step(self, n_children, parent_node): 
        '''
        It generates children nodes by computing new asset values and 
        the probabilities of each new node. Stock prices following 
        Geometric Brownian Motion are generated until a no arbitrage 
        setting is found. If the market is arbitrage free, option prices 
        (using Black and Scholes formula) and cash new values are computed.
        '''

        remaining_times = parent_node['remaining_times']-1 # remaining times of children nodes
        parent_stock_prices = parent_node['obs'][1:self.n_shares+1] 
        parent_cash_price = parent_node['obs'][0]

        '''arb = True 
        counter = 0 
        # Simulate stock prices until no-arbitarge is found:
        while (arb == True) and (counter<100):
            counter += 1
            # In simulations settings (Geometric Brownian Motion): 
            # S(t+dt) = S(t) * exp((mu - 1/2*sigma**2) * dt + sigma * sqrt(dt) * Z)
            # where Z is a standard normal distribution
            
            # Generate correlated Brownian increments
            Increment = MultiStock.generate_BM_stock_increments(
                self.n_shares, 
                self.risk_free_rate,
                self.mu, self.sigma, self.corr, 
                self.dt, n_children,
                self.rnd_state,
                risk_free = False)
            
            # Find new stock prices using the Geometric Brownian Motion formula
            stock_prices = np.zeros((self.n_shares, n_children))
            for i in range(self.n_shares):
                for s in range(n_children):
                    stock_prices[i,s] = parent_stock_prices[i] * np.exp(Increment[i,s]) 

            arb = check_arbitrage_prices(stock_prices, parent_stock_prices, self.risk_free_rate, self.dt)
            if (arb == False):
                logging.info(f"No arbitrage solution found after {counter} iteration(s)")
            
        if counter >= 100:
            raise RuntimeError(f"No arbitrage solution NOT found after {counter} iteration(s)")
        else:
            probs = self.compute_probabilities(n_children, parent_stock_prices, stock_prices)'''
        
        # In simulations settings (Geometric Brownian Motion): 
        # S(t+dt) = S(t) * exp((mu - 1/2*sigma**2) * dt + sigma * sqrt(dt) * Z)
        # where Z is a standard normal distribution
        
        # Generate correlated Brownian increments
        Increment = MultiStock.generate_BM_stock_increments(
            self.n_shares, 
            self.risk_free_rate,
            self.mu, self.sigma, self.corr, 
            self.dt, n_children,
            self.rnd_state,
            risk_free = False)
        
        # Find new stock prices using the Geometric Brownian Motion formula
        stock_prices = np.zeros((self.n_shares, n_children))
        for i in range(self.n_shares):
            for s in range(n_children):
                stock_prices[i,s] = parent_stock_prices[i] * np.exp(Increment[i,s]) 
        
        stock_prices = make_arbitrage_free_states(parent_stock_prices, stock_prices, n_children)
        probs = self.compute_probabilities(n_children, parent_stock_prices, stock_prices)

        # Options values 
        option_prices = np.zeros((self.n_options, n_children))
        time_to_maturity = remaining_times * self.dt 
        if time_to_maturity != 0:
            for j, option in enumerate(self.option_list):
                underlying_value = stock_prices[option.underlying_index,:]
                option_prices[j,:] = option.BlackScholesPrice(underlying_value, time_to_maturity)
        else: 
            for j, option in enumerate(self.option_list):
                underlying_value = stock_prices[option.underlying_index,:]
                option_prices[j,:] = option.get_payoff(underlying_value)

        # Cash value
        cash_price = parent_cash_price * np.exp(self.risk_free_rate*self.dt) * np.ones(shape=n_children)

        prices = np.vstack((cash_price, stock_prices, option_prices))
        
        return probs, prices
    
    
    def _MM_objective(self, p, *args): 
        '''Objective function of the Moment Matching model. p is the 
        vector of decision variables (node probabilities).'''
        
        parent_stock_prices, stock_prices, n_children = args 
        # Get returns from prices:
        returns = np.zeros((len(parent_stock_prices), n_children))
        for i in range(len(parent_stock_prices)):
            returns[i, :] = np.log(stock_prices[i, :] / parent_stock_prices[i])

        # Statistical moments of the tree
        tree_mean = mean(returns, p) 
        tree_moment2 = second_moment(returns, p) 
        tree_cor = correlation(returns, p)

        # Geometric Brownian Motion moments
        true_moment1 = np.zeros(self.n_shares)
        true_moment2 = np.zeros(self.n_shares)
        for j in range(self.n_shares):
            true_moment1[j] = self.moments(dynamics='BS', number=1, underlying_index=j)
            true_moment2[j] = self.moments(dynamics='BS', number=2, underlying_index=j)

        # The objective function is the sum the norms of the difference between each expected moment
        # and the moment of the generated tree
        sqdiff = (np.linalg.norm(true_moment1 - tree_mean, 2) + 
                  np.linalg.norm(true_moment2 - tree_moment2, 2) + 
                  np.linalg.norm(self.corr - tree_cor, 1))
        
        return sqdiff
    

    def _MM_constraint(self, p):
        '''Probs sum up to one.'''

        return np.sum(p) - 1


    def compute_probabilities(self, n_children, parent_stock_prices, stock_prices):
        '''
        Compute probabilities associated to children nodes via moment matching.
        '''

        # Define initial solution: uniform nodes probabilities
        p_init = (1 / n_children) * np.ones(n_children)
                
        # Define probabilities bounds
        bounds = [(1/(n_children*10), 1)] * (n_children) 
        
        # Define constraints
        constraints = [{'type': 'eq', 'fun': self._MM_constraint}]

        args = (parent_stock_prices, stock_prices, n_children)

        # Run optimization
        res = optimize.minimize(self._MM_objective, p_init, method='SLSQP', bounds=bounds, args=args, constraints=constraints, options={'maxiter': 5000})
        
        # Store the solution
        probabilities = res.x
        
        return probabilities
    
    
    def moments(self, dynamics: str, number: int, underlying_index: int):
        '''
        Get the exact moment (number=1,2,...) of the specified dynamics (e.g., BS, VG).
        '''
        j = underlying_index

        if dynamics == 'BS':
            if number == 1: moment = (self.mu[j] - 1/2 * self.sigma[j]**2) * self.dt
            if number == 2: moment = self.sigma[j]**2 * self.dt + (self.mu[j] * self.dt)**2
        
        if dynamics == 'VG': 
            if number == 1: moment = self.mu[j] * self.dt
            if number == 2: moment = (self.sigma[j]**2 + self.mu[j]**2 * self.nu[j]) * self.dt + (self.mu[j] * self.dt)**2
        
        return moment
    