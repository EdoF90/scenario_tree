# -*- coding: utf-8 -*-
import logging
import numpy as np
import yfinance as yf
import gurobipy as gp
from scipy import stats
from gurobipy import GRB
from scipy import optimize
from .stochModel import StochModel
from .checkarbitrage import check_arbitrage_prices
from .calculatemoments import mean, std, correlation

''' 
BrownianMotionForHedging: stochastic model used to simulate stock price dynamics 
uder the Geometric Brownian Motion model.
simulate_one_time_step: for each parent node in scenario tree, it generates children 
    nodes by computing new asset values and the probabilities of each new node.
    Stock prices following Geometric Brownian Motion are generated until a no arbitrage 
    setting is found. If the market is arbitrage free, option prices (using Black and Scholes formula)
    and cash new values are computed.
'''

class BrownianMotionForHedging(StochModel):

    def __init__(self, 
                 sim_setting, 
                 option_list, 
                 dt, mu, 
                 sigma, rho, 
                 skew, kur, 
                 rnd_state): 
        
        super().__init__(sim_setting)
        self.dt = dt 
        self.n_options = len(option_list)
        self.option_list = option_list
        self.mu = mu
        self.sigma = sigma
        self.corr = rho 
        self.skew = skew
        self.kur = kur
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
            # probs = 1/n_children * np.ones(n_children) #  uniform probabilities ? 
            probs = self.compute_probabilities(n_children, parent_stock_prices, stock_prices)

        # Options values
        option_prices = np.zeros((self.n_options, n_children))
        for j in range(self.n_shares):
            S0 = stock_prices[j,:]
            time_to_maturity = remaining_times * self.dt 
            if time_to_maturity != 0:
                # hedging options are assumed to be of European type
                option_prices[j,:] = self.option_list[j].BlackScholesPrice(S0, time_to_maturity)
                option_prices[j+self.n_shares,:] = self.option_list[j+self.n_shares].BlackScholesPrice(S0, time_to_maturity)
            else:
                option_prices[j,:] = self.option_list[j].get_payoff(S0)
                option_prices[j+self.n_shares,:] = self.option_list[j+self.n_shares].get_payoff(S0)

        # Cash value
        cash_price = parent_cash_price * np.exp(self.option_list[0].risk_free_rate*self.dt) * np.ones(shape=n_children)

        prices = np.vstack((cash_price, stock_prices, option_prices))
        
        return probs, prices
    
    
    def _MM_objective(self, p, *args): # objective function of the MM model
        # p is the vector of decision variables (node probabilities)
        
        parent_stock_prices, stock_prices, n_children = args 
        # Get returns from prices:
        returns = np.zeros((len(parent_stock_prices), n_children))
        for i in range(len(parent_stock_prices)):
            returns[i, :] = np.log(stock_prices[i, :] / parent_stock_prices[i])

        # Following lines calculate the statistical moments of the tree
        tree_mean = mean(returns, p)
        tree_std = std(returns, p) 
        tree_cor = correlation(returns, p)

        true_moment1 = np.zeros(self.n_shares)
        true_moment2 = np.zeros(self.n_shares)
        for j in range(self.n_shares):
            true_moment1[j] = self.moments(dynamics='BS', number=1, underlying_index=j)
            true_moment2[j] = self.moments(dynamics='BS', number=2, underlying_index=j)

        # The objective function is the squared difference among the expexted moments and the moments underlying the generated tree
        sqdiff = (np.linalg.norm(true_moment1 - tree_mean, 2) + 
                  np.linalg.norm(true_moment2 - tree_std, 2) + 
                  np.linalg.norm(self.corr - tree_cor, 1))
        
        return sqdiff
    

    def _MM_constraint(self, p):
        # Probs sum up to one
        return np.sum(p) - 1


    def compute_probabilities(self, n_children, parent_stock_prices, stock_prices):
        '''
        Compute the vector of probabilities, associated to the next nodes,
        that best approximate the continuous process, according to the generated states.
        This is obtained via moment matching.
        Refer to Hoyland (2001) for a similar method.
        '''

        # Define initial solution: equal probabilities for each node
        initial_solution = []
        p_init = (1 / n_children) * np.ones(n_children)
        initial_solution.append(p_init)
                
        # Define bounds
        bounds = [(0.05, 0.4)] * (n_children) # bounds for probabilities to avoid vanishing probabilities
        
        '''
        mean_bound = np.mean(self.mu)
        std_bound = np.max(self.sigma)
        lb = mean_bound - 3*std_bound
        ub = mean_bound + 3*std_bound
        '''

        # Define constraints
        constraints = [{'type': 'eq', 'fun': self._MM_constraint}]

        args = (parent_stock_prices, stock_prices, n_children)

        # Running optimization
        res = optimize.minimize(self._MM_objective, initial_solution, method='SLSQP', args=args, bounds=bounds, constraints=constraints, options={'maxiter': 5000})
        probabilities = res.x
        
        return probabilities
    
    
    def moments(self, dynamics: str, number: int, underlying_index: int):
        '''
        Get the exact moment (number=1,2,...) of a certain dynamics (e.g., VG).
        '''
        j = underlying_index

        if dynamics == 'BS':
            if number == 1: moment = self.mu[j] * self.dt
            if number == 2: moment = self.sigma[j]**2 * self.dt + (self.mu[j] * self.dt)**2
        
        if dynamics == 'VG':
            if number == 1: moment = self.mu[j] * self.dt
            if number == 2: moment = (self.sigma[j]**2 + self.mu[j]**2 * self.nu[j]) * self.dt + (self.mu[j] * self.dt)**2
        
        return moment
