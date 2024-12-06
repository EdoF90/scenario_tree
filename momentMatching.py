import logging
import time
import numpy as np
import yfinance as yf
from scipy import stats
from scipy import optimize
from .stochModel import StochModel
from .checkarbitrage import check_arbitrage_prices
from .calculatemoments import mean, std, skewness, kurtosis, correlation


class MomentMatching(StochModel): # Instance of the abstract class StochModel

    ''' 
    Stochastic model used to simulate stock price dynamics and children probabilities by 
    matching simulated properties (first, third and fourth moments, std, correlation,...) 
    with historical ones.
    '''


    def __init__(self, sim_setting):
        self.tickers = sim_setting['tickers']
        self.n_shares = len(self.tickers)
        self.start_date = sim_setting["start"]
        self.end_date = sim_setting["end"]

        self.ExpectedMomentsEstimate( # definition of the expected moments to match
            start = self.start_date,
            end = self.end_date
        ) 
    
    def ExpectedMomentsEstimate(self, start, end): 
        ''' Historical moments estimate.'''

        hist_prices = yf.download(
            self.tickers, 
            start=start,
            end=end
            )['Close']
        monthly_prices = hist_prices.resample('ME').last() 
        log_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()

        self.exp_mean = log_returns.mean().values
        self.exp_std = log_returns.std().values
        self.exp_skew = log_returns.apply(
            lambda x: stats.skew(x, bias=False)
            ).values
        self.exp_kur = log_returns.apply(
            lambda x: stats.kurtosis(x, bias=False, fisher=False)
            ).values
        self.exp_cor = log_returns.corr().values
    
    
    def _objective(self, y): 
        '''Objective function of the MM model. y is the vector of decision variables: 
        the first n_children entries are node probabilities, the remaining are log-returns.'''

        p = y[:self.n_children] # probabilities
        x = y[self.n_children:] # log-returns
        # log-returns in matrix form (row: shares , column: scenario)
        x_matrix = x.reshape((self.n_shares, self.n_children)) 
        # Following lines calculate the statistical moments of the tree
        tree_mean = mean(x_matrix, p)
        tree_std = std(x_matrix, p) 
        tree_skew = skewness(x_matrix, p)
        tree_kurt = kurtosis(x_matrix, p)
        tree_cor = correlation(x_matrix, p)

        # The objective function is the squared difference between the expexted moments 
        # and the moments of the generated tree
        sqdiff = (np.linalg.norm(self.exp_mean - tree_mean, 2) + np.linalg.norm(self.exp_std - tree_std, 2) + 
                np.linalg.norm(self.exp_skew - tree_skew, 2) + np.linalg.norm(self.exp_kur - tree_kurt, 2) +
                np.linalg.norm(self.exp_cor - tree_cor, 1))
        
        return sqdiff
    
    
    def _constraint(self, y):
        '''Probs sum up to one.'''

        p = y[:self.n_children]
        return np.sum(p) - 1
    
    
    def solve(self):
        '''Solve an SLSQP problem to find probabilities and values.'''

        # Define an initial solution: uniform nodes probabilities, log-returns 
        # sampled from a Normal distribution
        initial_solution_parts = []
        p_init = (1 / self.n_children) * np.ones(self.n_children)
        initial_solution_parts.append(p_init)
        for a in range(self.n_shares):
            part = np.abs(np.random.normal(loc=self.exp_mean[a], scale=self.exp_std[a], size=self.n_children))
            initial_solution_parts.append(part)
                
        initial_solution = np.concatenate(initial_solution_parts)
        
        # Define probabilities bounds 
        bounds_p= [(0, 1)] * (self.n_children) 
        
        # Define log-returns bounds 
        bounds_x = [(None, None)] * (self.n_shares * self.n_children)

        bounds = bounds_p + bounds_x
        
        # Define constraints
        constraints = [{'type': 'eq', 'fun': self._constraint}]

        # Run optimization
        res = optimize.minimize(self._objective, initial_solution, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 5000})
        
        # Store the solution
        p_res = res.x[:self.n_children] # probabilities
        x_mat = res.x[self.n_children:].reshape((self.n_shares, self.n_children)) # values 
        
        return p_res, x_mat, res.fun
    

    def simulate_one_time_step(self, n_children, parent_node):
        ''' 
        It generates children nodes by computing new asset values and 
        the probabilities of each new node. Stock prices are generated 
        until a no arbitrage setting is found.
        '''
        
        self.n_children = n_children # set the number of nodes to generate
        
        # Inizialization
        arb = True
        counter = 0
        flag = True
        startt = time.time()

        # Main loop: keeps solving the MM model until an arbitrage-free good quality 
        # solution is found (or until the maximum number of iterations is reached)
        while flag and (counter < 100):
            counter += 1

            probs, returns, fun = self.solve() # solve the MM model

            # Get prices from returns
            prices = np.zeros((len(parent_node), n_children))
            for i in range(len(parent_node)):
                for s in range(n_children):
                    prices[i,s] = parent_node[i] * np.exp(returns[i,s])
            
            # Check if there is an arbitrage opportunity
            arb = check_arbitrage_prices(prices, parent_node)

            # If an arbitrage-free and good quality solution is found, then log some info, 
            # add the generated nodes to the tree and break the loop
            if (arb == False) and (fun <= 1):
                flag = False
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
            raise RuntimeError("Good quality arbitrage-free scenarios not found after 100 attempts")
        else:
            endt = time.time()
            logging.info(f"Computational time to build the tree:{endt - startt} seconds")
            return probs, prices # return probababilities and prices to add nodes to the tree
