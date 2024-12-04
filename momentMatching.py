# -*- coding: utf-8 -*-
import logging
import numpy as np
import yfinance as yf
from scipy import stats
from scipy import optimize
from .stochModel import StochModel
from .checkarbitrage import check_arbitrage_prices
from .calculatemoments import mean, std, skewness, kurtosis, correlation


class MomentMatching(StochModel):

    def __init__(self, sim_setting):
        super().__init__(sim_setting)
        self.n_children = 0
        self.parent_node = []
        self.ExpectedMomentsEstimate(
            sim_setting['start'],
            sim_setting['end']
        )
    
    def ExpectedMomentsEstimate(self, start, end):
        data = yf.download(
            self.tickers,
            start= start,
            end= end
        )['Adj Close']
        hist_prices = data.dropna()
        #returns = data.pct_change().dropna()
        self.exp_mean = hist_prices.mean().values
        self.exp_std = hist_prices.std().values
        self.exp_skew = hist_prices.apply(
            lambda x: stats.skew(x, bias=False)
        ).values
        self.exp_kur = hist_prices.apply(
            lambda x: stats.kurtosis(x, bias=False, fisher=False)
        ).values
        self.exp_cor = hist_prices.corr().values
    
    def set_n_children(self, n_children):
        self.n_children = n_children

    def set_parent_node(self, parent_node):
        self.parent_node = []
        self.parent_node = parent_node
    
    def _objective(self, y):
    # def _objective(y, n_children, ):
        p = y[:self.n_children]
        nu = y[self.n_children:2*self.n_children]
        x = y[2*self.n_children:]
        x_matrix = x.reshape((self.n_shares, self.n_children))
        tree_mean = mean(x_matrix, p)
        tree_std = std(x_matrix, p)
        tree_skew = skewness(x_matrix, p)
        tree_kurt = kurtosis(x_matrix, p)
        tree_cor = correlation(x_matrix, p)
        sqdiff = (
            np.linalg.norm(self.exp_mean - tree_mean, 2) +
            np.linalg.norm(self.exp_std - tree_std, 2) + 
            np.linalg.norm(self.exp_skew - tree_skew, 2) +
            np.linalg.norm(self.exp_kur - tree_kurt, 2) +
            np.linalg.norm(self.exp_cor - tree_cor, 1)
        )
        
        return sqdiff
    
    def _constraint(self, y):
        p = y[:self.n_children]
        nu = y[self.n_children:2*self.n_children]
        x = y[2*self.n_children:]
        x_matrix = x.reshape((self.n_shares, self.n_children))
        constraints=[]
        # prob sum to one
        constraints.append(np.sum(p) - 1)
        mart_expvalue = np.zeros(self.n_shares)
        for i in range(self.n_shares):
            for s in range(self.n_children):
                mart_expvalue[i] += nu[s] * x_matrix[i,s]
            constraints.append(mart_expvalue[i]-self.parent_node[i])
        return constraints
    
    def solve(self):
        # Define initial solution
        initial_solution_parts = []
        p_init = (1 / self.n_children) * np.ones(self.n_children)
        initial_solution_parts.append(p_init)
        nu_init = (1 / self.n_children) * np.ones(self.n_children)
        initial_solution_parts.append(nu_init)
        for a in range(self.n_shares):
            part = np.abs(np.random.normal(loc=self.exp_mean[a], scale=self.exp_std[a], size=self.n_children))
            initial_solution_parts.append(part)
                
        initial_solution = np.concatenate(initial_solution_parts)

        # Define bounds
        bounds = [(0, np.inf)]*(self.n_children*(2+self.n_shares))

        # Define constraints
        constraints = [{'type': 'eq', 'fun': self._constraint}]

        # Running optimization
        res = optimize.minimize(
            self._objective,
            initial_solution,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': 10000
            }
        )
        p_res = res.x[:self.n_children]
        nu_res = res.x[self.n_children:2*self.n_children]
        x_res = res.x[2*self.n_children:]
        x_mat = x_res.reshape((self.n_shares, self.n_children))
        return p_res, x_mat, nu_res, res.fun

    # TODO: generalize to a non stationary process
    def simulate_one_time_step(self, n_children, parent_node):      
        self.set_n_children(n_children)
        self.set_parent_node(parent_node)
        arb = True
        counter = 0
        while arb and (counter < 100):
            probs, prices, martprob, fun = self.solve()
            counter += 1
            arb = check_arbitrage_prices(prices, parent_node)
            if (arb == False) and (fun<10):
                tree_mean = mean(prices, probs)
                tree_std = std(prices, probs)
                tree_skew = skewness(prices, probs)
                tree_kurt = kurtosis(prices, probs)
                tree_cor = correlation(prices, probs)
                logging.info(f"No arbitrage solution found after {counter} iteration(s)")
                logging.info(f"Objective function value: {fun}")
                logging.info(f"Expected vs Generated Moments:")
                logging.info(f"Mean: Expected = {self.exp_mean}, Generated = {tree_mean}, Exp-Gen = {self.exp_mean - tree_mean}")
                logging.info(f"Std Dev: Expected = {self.exp_std}, Generated = {tree_std}, Exp-Gen = {self.exp_std - tree_std}")
                logging.info(f"Skewness: Expected = {self.exp_skew}, Generated = {tree_skew}, Exp-Gen = {self.exp_skew - tree_skew}")
                logging.info(f"Kurtosis: Expected = {self.exp_kur}, Generated = {tree_kurt}, Exp-Gen = {self.exp_kur - tree_kurt}")
                logging.info(f"Correlation: Expected = {self.exp_cor}, Generated = {tree_cor}, Exp-Gen = {self.exp_cor - tree_cor}")

        if counter >= 100:
            raise RuntimeError("Good quality arbitrage-free scenarios not found")
        else:
            return probs, prices
                
    #TODO: Need to properly define this function
    def simulate_all_horizon(self, time_horizon):
        return 0
