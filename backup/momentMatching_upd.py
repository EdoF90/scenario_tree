import yfinance as yf
from ..stochModel import StochModel
import numpy as np
from scipy import stats
from scipy import optimize
from ..calculatemoments import mean, std, skewness, kurtosis, correlation
from ..checkarbitrage import check_arbitrage_prices
import logging
import os

class MomentMatching(StochModel):

    def __init__(self, sim_setting):
        super().__init__(sim_setting)
        # self.n_children = 0
        self.ExpectedMomentsEstimate(
            start = sim_setting["start"],
            end = sim_setting["end"]
        )
    
    def ExpectedMomentsEstimate(self, start, end):
        hist_prices = yf.download(
            self.tickers, 
            start=start,
            end=end
            )['Adj Close']
        log_returns = np.log(hist_prices / hist_prices.shift(1)).dropna()

        self.exp_mean = log_returns.mean().values
        self.exp_std = log_returns.std().values
        self.exp_skew = log_returns.apply(lambda x: stats.skew(x, bias=False)).values
        self.exp_kur = log_returns.apply(lambda x: stats.kurtosis(x, bias=False, fisher=False)).values
        self.exp_cor = log_returns.corr().values
    
    '''
    def update_expectedstd(self, parent_node):
        for i in range(self.n_shares):
            self.exp_std[i] = self.VC[i] * abs(parent_node[i] - self.exp_mean[i]) + (1-self.VC[i]) * (self.exp_std[i])
    
    def update_expectedmean(self, parent_node):
        for i in range(self.n_shares):
            self.exp_mean[i] = self.risk_free_return + self.RP[i] * self.exp_std[i]
    
    def set_n_children(self, n_children):
        self.n_children = n_children
    '''
    def _objective(y, n_shares, n_children, exp_mean, exp_std, exp_skew, exp_kur, exp_cor):
        p = y[:n_children]
        x = y[n_children:]
        x_matrix = x.reshape((n_shares, n_children))
        tree_mean = mean(x_matrix, p)
        tree_std = std(x_matrix, p)
        tree_skew = skewness(x_matrix, p)
        tree_kurt = kurtosis(x_matrix, p)
        tree_cor = correlation(x_matrix, p)
        sqdiff = (np.linalg.norm(exp_mean - tree_mean, 2) + np.linalg.norm(exp_std - tree_std, 2) + 
                np.linalg.norm(exp_skew - tree_skew, 2) + np.linalg.norm(exp_kur - tree_kurt, 2) +
                np.linalg.norm(exp_cor - tree_cor, 1))
        
        return sqdiff
   
    def _constraint(y, n_children):
        # probs sum up to one
        p = y[:n_children]
        return np.sum(p) - 1
    
    def solve(self, n_children):
        # Define initial solution
        initial_solution_parts = []
        p_init = (1 / n_children) * np.ones(n_children)
        initial_solution_parts.append(p_init)
        for a in range(self.n_shares):
            part = np.abs(np.random.normal(loc=self.exp_mean[a], scale=self.exp_std[a], size= n_children))
            initial_solution_parts.append(part)
                
        initial_solution = np.concatenate(initial_solution_parts)

        # Define bounds
        bounds_p= [(1e-3, np.inf)] * (n_children)
        mean_bound = np.mean(self.exp_mean)
        std_bound = np.max(self.exp_std)
        lb = mean_bound - 3*std_bound
        ub = mean_bound + 3*std_bound
        bounds_x = [(lb, ub)] * (self.n_shares * n_children)
        # bounds_x = [(None, None)] * (self.n_shares * self.n_children)
        bounds = bounds_p + bounds_x
        
        # Define objective function and constraints
        obj_fun = lambda x: self._objective(
            x, 
            self.n_shares, 
            n_children, 
            self.exp_mean,
            self.exp_std, 
            self.exp_skew, 
            self.exp_kur,
            self.exp_cor)
        
        #constr = lambda x: self._constraint(x, n_children)
        constraints = [{'type': 'eq', 'fun': lambda x: self._constraint(x, n_children=n_children)}]
        # Running optimization
        res = optimize.minimize(obj_fun, initial_solution, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 5000})
        p_res = res.x[:n_children]
        x_res = res.x[n_children:]
        x_mat = x_res.reshape((self.n_shares, n_children))
        return p_res, x_mat, res.fun

    def simulate_one_time_step(self, n_children, parent_node):
        '''
        if period > 1:
            self.update_expectedstd(parent_node)
            self.update_expectedmean(parent_node)
        '''        
        arb = True
        counter = 0
        while arb and (counter < 100):
            counter += 1
            probs, returns, fun = self.solve(n_children)
            prices = np.zeros((len(parent_node), n_children))
            for i in range(len(parent_node)):
                for s in range(n_children):
                    prices[i,s] = parent_node[i] * np.exp(returns[i,s])
            arb = check_arbitrage_prices(prices, parent_node)
            if (arb == False) and (fun <= 5):
                tree_mean = mean(returns, probs)
                tree_std = std(returns, probs)
                tree_skew = skewness(returns, probs)
                tree_kurt = kurtosis(returns, probs)
                tree_cor = correlation(returns, probs)
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