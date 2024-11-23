# -*- coding: utf-8 -*-
import logging
import numpy as np
import yfinance as yf
import gurobipy as gp
from scipy import stats
from gurobipy import GRB
from .stochModel import StochModel
from .checkarbitrage import check_arbitrage_prices

''' 
BrownianMotionForHedging: stochastic model used to simulate stock price dynamics 
uder the Geometric Brownian Motion model.
simulate_one_time_step: for each parent node in scenario tree, it generates children 
    nodes by computing new asset values and the probabilities of each new node.
    Stock prices following Geometric Brownian Motion are generated until a no arbitrage 
    setting is found. If the market is arbitrage free, option prices (using Black and Scholes formula)
    and cash new values are computed.
'''

class BrownianMotionForHedging_Gurobi(StochModel):

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
    

    def compute_probabilities(self, n_children, parent_stock_prices, stock_prices):
        '''
        Compute the vector of probabilities, associated to the next nodes,
        that best approximate the continuous process, according to the generated states.
        This is obtained via moment matching (only 1st and 2nd by now).
        Refer to Hoyland (2001) for a similar method.
        '''
        M = gp.Model("Get probabilities that best approximate the continuous process")
        p = []
        for i in range(n_children):
            p.append(M.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='p'+str(i+1)))
        M.addConstr(np.sum(p) == 1, name='sum=1')
        
        Diff1 = np.zeros(0)
        Diff2 = np.zeros(0)
        Diff3 = np.zeros(0)

        x = np.zeros((self.n_shares, n_children))
        for j in range(self.n_shares):
            x[j,:] = np.log(stock_prices[j,:] / parent_stock_prices[j])

        for j in range(self.n_shares):
            true_moment1 = self.moments(dynamics='BS', number=1, underlying_index=j)
            popu_moment1 = np.log(stock_prices[j,:] / parent_stock_prices[j]) @ np.array(p)
            diff1 = (true_moment1 - popu_moment1)**2
            true_moment2 = self.moments(dynamics='BS', number=2, underlying_index=j)
            popu_moment2 = np.log(stock_prices[j,:] / parent_stock_prices[j])**2 @ np.array(p)
            diff2 = (true_moment2 - popu_moment2)**2
            Diff1 = np.hstack((Diff1, diff1))
            Diff2 = np.hstack((Diff2, diff2)) 

            true_property = self.moments(dynamics='BS', number=-1, underlying_index=j)
            for index, i in enumerate(range(j+1, self.n_shares)):                
                popu_property = (np.log(stock_prices[j,:] / parent_stock_prices[j]) * np.log(stock_prices[i,:] / parent_stock_prices[i])) @ np.array(p)
                diff3 = (true_property[index] - popu_property)**2 #TODO  provato con il quadrato della differenza, è ok? 
                Diff3 = np.hstack((Diff3, diff3))

        M.setObjective(np.sum(Diff1) + np.sum(Diff2) + np.sum(Diff3), GRB.MINIMIZE) 
        M.Params.LogToConsole = 0  # ...avoid printing all info with m.optimize()
        M.optimize()
        probabilities = np.zeros(n_children)
        for i in range(n_children):
            probabilities[i] = M.getVars()[i].X
 
        return probabilities
    
    
    def moments(self, dynamics: str, number: int, underlying_index: int):
        '''
        Get the exact moment (number=1,2,...) of a certain dynamics (e.g., VG).
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
 