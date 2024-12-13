# -*- coding: utf-8 -*-
import logging
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from assets import MultiStock
from .stochModel import StochModel
from .checkarbitrage import check_arbitrage_prices

class BrownianMotionForHedging_Gurobi(StochModel):
    '''
    Stochastic model used to simulate stock price dynamics 
    under the Geometric Brownian Motion model. The moment matching 
    optimization problem used to find children nodes probabilities 
    is solved with Gurobi. The matched properties are: first and 
    second moment, E[XY]. Options are then priced via Black&Scholes 
    formula.
    '''

    def __init__(self, 
                 sim_setting, 
                 option_list,
                 dt, mu, 
                 sigma, rho, 
                 skew, kur, 
                 rnd_state): 
        
        self.n_shares = len(sim_setting['tickers'])
        self.dt = dt 
        self.n_options = len(option_list)
        self.option_list = option_list
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
        parent_stock_prices= parent_node['obs'][1:self.n_shares+1] 
        parent_cash_price = parent_node['obs'][0]

        arb = True
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
                self.option_list[0].risk_free_rate,
                self.mu, self.sigma, self.corr, 
                self.dt, n_children,
                self.rnd_state,
                risk_free = False)

            # Find new stock prices using the Geometric Brownian Motion formula
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
        cash_price = parent_cash_price * np.exp(self.option_list[0].risk_free_rate*self.dt) * np.ones(shape=n_children)

        prices = np.vstack((cash_price, stock_prices, option_prices))
        
        return probs, prices
    

    def compute_probabilities(self, n_children, parent_stock_prices, stock_prices):
        '''
        Compute probabilities associated to children nodes via moment matching.
        '''
          
        M = gp.Model("Get probabilities that best approximate the continuous process")
        p = []
        for i in range(n_children): # decision variables: probabilities
            p.append(M.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='p'+str(i+1)))
        M.addConstr(np.sum(p) == 1, name='sum=1') # probs sum up to 1
        
        Diff1 = np.zeros(0)
        Diff2 = np.zeros(0)
        Diff3 = np.zeros(0)

        # Matrix of log-returns
        x = np.zeros((self.n_shares, n_children))
        for j in range(self.n_shares):
            x[j,:] = np.log(stock_prices[j,:] / parent_stock_prices[j])

        for j in range(self.n_shares):
            # First moment
            true_moment1 = self.moments(dynamics='BS', number=1, underlying_index=j)
            tree_moment1 = np.log(stock_prices[j,:] / parent_stock_prices[j]) @ np.array(p)
            diff1 = (true_moment1 - tree_moment1)**2
            Diff1 = np.hstack((Diff1, diff1))
            
            # Second moment
            true_moment2 = self.moments(dynamics='BS', number=2, underlying_index=j)
            tree_moment2 = np.log(stock_prices[j,:] / parent_stock_prices[j])**2 @ np.array(p)
            diff2 = (true_moment2 - tree_moment2)**2
            Diff2 = np.hstack((Diff2, diff2)) 

            # E[XY]
            true_property = self.moments(dynamics='BS', number=-1, underlying_index=j)
            for index, i in enumerate(range(j+1, self.n_shares)):                
                tree_property = (np.log(stock_prices[j,:] / parent_stock_prices[j]) * np.log(stock_prices[i,:] / parent_stock_prices[i])) @ np.array(p)
                diff3 = (true_property[index] - tree_property)**2
                Diff3 = np.hstack((Diff3, diff3))

        
        # The objective function is the sum of the squared difference between each expected moment
        # and the moment of the generated tree
        M.setObjective(np.sum(Diff1) + np.sum(Diff2) + np.sum(Diff3), GRB.MINIMIZE) 
        M.Params.LogToConsole = 0  # avoid printing all info with m.optimize()
        M.optimize()

        # Store solution
        probabilities = np.zeros(n_children)
        for i in range(n_children):
            probabilities[i] = M.getVars()[i].X
 
        return probabilities
    
    
    def moments(self, dynamics: str, number: int, underlying_index: int):
        '''
        Get the exact moment (number=-1,1,2,...) of the specified dynamics (e.g., BS, VG).
        '''
        j = underlying_index

        if dynamics == 'BS':
            if number == 1: moment = self.mu[j] * self.dt
            if number == 2: moment = self.sigma[j]**2 * self.dt + (self.mu[j] * self.dt)**2
            if number == -1: # E[j i], for i > j 
                moment = []
                for i in range(j+1, self.n_shares): 
                    moment.append((self.sigma[j]*self.sigma[i]*self.dt) * self.corr[i,j] + self.mu[j]*self.dt*self.mu[i]*self.dt)
                moment = np.array(moment)     
        
        if dynamics == 'VG':
            if number == 1: moment = self.mu[j] * self.dt
            if number == 2: moment = (self.sigma[j]**2 + self.mu[j]**2 * self.nu[j]) * self.dt + (self.mu[j] * self.dt)**2
        
        return moment
 