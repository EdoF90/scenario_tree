import logging
import time
import numpy as np
import yfinance as yf
from scipy import stats
from scipy import optimize
from .stochModel import StochModel
from .checkarbitrage import check_arbitrage_prices
from .calculatemoments import mean, std, skewness, kurtosis, correlation


class MomentMatchingForHedging(StochModel): # Instance of the abstract class StochModel

    def __init__(self, 
                 sim_setting, 
                 option_list,
                 dt, mu, sigma, 
                 rho, skew, # kur,
                 rnd_state):
        super().__init__(sim_setting)
        self.n_options = len(option_list)
        self.option_list = option_list
        self.dt = dt
        self.mu = mu * self.dt 
        self.sigma = sigma * np.sqrt(self.dt)
        self.corr = rho
        self.skew = skew #TODO: is it correct or is it to be modified? 
        #self.kur = kur #TODO: is it correct or is it to be modified? 
        self.rnd_state = rnd_state
        

    
    def set_n_children(self, n_children): # set the number of children to generate
        self.n_children = n_children
    

    def _objective(self, y): # objective function of the MM model
        # y is the vector of decision variables: the first n_children entries are node probabilities, the remaining are log-returns
        p = y[:self.n_children] # probabilities
        x = y[self.n_children:] # log-returns
        x_matrix = x.reshape((self.n_shares, self.n_children)) # log-returns in matrix form (row: shares - column: scenario)
        # Following lines calculate the statistical moments of the tree
        tree_mean = mean(x_matrix, p)
        tree_std = std(x_matrix, p) 
        tree_skew = skewness(x_matrix, p)
        tree_kurt = kurtosis(x_matrix, p)
        tree_cor = correlation(x_matrix, p)

        # The objective function is the squared difference among the expexted moments and the moments underlying the generated tree
        sqdiff = (np.linalg.norm(self.mu - tree_mean, 2) + np.linalg.norm(self.sigma - tree_std, 2) + 
                np.linalg.norm(self.skew - tree_skew, 2) + #+ np.linalg.norm(self.kur - tree_kurt, 2) +
                np.linalg.norm(self.corr - tree_cor, 1))
        
        return sqdiff
    
    
    def _constraint(self, y):
        # Probs sum up to one
        p = y[:self.n_children]
        return np.sum(p) - 1
    

    def solve(self):
        # Define initial solution: equal proabibilities for each node, log-returns sampled from a Normal distribution
        initial_solution_parts = []
        p_init = (1 / self.n_children) * np.ones(self.n_children)
        initial_solution_parts.append(p_init)
        for a in range(self.n_shares):
            part = np.abs(self.rnd_state.normal(loc=self.mu[a], scale=self.sigma[a], size=self.n_children))
            initial_solution_parts.append(part)
                
        initial_solution = np.concatenate(initial_solution_parts)

        # Define bounds
        bounds_p= [(0.05, 0.4)] * (self.n_children) # bounds for probabilities to avoid vanishing probabilities
        '''
        mean_bound = np.mean(self.mu)
        std_bound = np.max(self.sigma)
        lb = mean_bound - 3*std_bound
        ub = mean_bound + 3*std_bound
        bounds_x = [(lb, ub)] * (self.n_shares * self.n_children)
        '''
        bounds_x = [(None, None)] * (self.n_shares * self.n_children) # No bounds for log-returns, if you want to bound them uncomment lines 84-88

        bounds = bounds_p + bounds_x
        
        # Define constraints
        constraints = [{'type': 'eq', 'fun': self._constraint}]

        # Running optimization
        res = optimize.minimize(self._objective, initial_solution, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 5000})
        p_res = res.x[:self.n_children]
        x_res = res.x[self.n_children:]
        x_mat = x_res.reshape((self.n_shares, self.n_children))

        return p_res, x_mat, res.fun
    

    def simulate_one_time_step(self, n_children, parent_node, remaining_times):
        
        parent_stock_prices= parent_node[1:self.n_shares+1] 
        parent_cash_price = parent_node[0]

        self.set_n_children(n_children) # set the number of nodes to generate
        # Inizialization
        arb = True
        counter = 0
        flag = True
        start_t = time.time()

        # Main loop: keeps solving the MM model until an arbitrage-free good quality solution is found (or until the maximum number of iterations is reached)
        while flag and (counter < 100):
            counter += 1

            probs, returns, fun = self.solve() # solve the MM model

            # Get prices from returns
            stock_prices = np.zeros((len(parent_stock_prices), n_children))
            for i in range(len(parent_stock_prices)):
                for s in range(n_children):
                    stock_prices[i,s] = parent_stock_prices[i] * np.exp(returns[i,s])
            
            # Check if there is an arbitrage opportunity
            arb = check_arbitrage_prices(stock_prices, parent_stock_prices)
            #arb = False
            # If an arbitrage-free and good quality solution is found, then log some info, add the generated nodes to the tree and break the loop
            '''tree_mean = mean(returns, probs)
            tree_std = std(returns, probs)
            tree_skew = skewness(returns, probs)
            tree_kurt = kurtosis(returns, probs)
            tree_cor = correlation(returns, probs)
            print(f"No arbitrage solution found after {counter} iteration(s)")
            print(f"Objective function value: {fun}")
            print(f"Expected vs Generated Moments:")
            print(f"Mean: Expected = {self.mu}, Generated = {tree_mean}, Exp-Gen = {self.mu - tree_mean}")
            print(f"Std Dev: Expected = {self.sigma}, Generated = {tree_std}, Exp-Gen = {self.sigma - tree_std}")
            print(f"Skewness: Expected = {self.skew}, Generated = {tree_skew}, Exp-Gen = {self.skew - tree_skew}")
            #print(f"Kurtosis: Expected = {self.kur}, Generated = {tree_kurt}, Exp-Gen = {self.kur - tree_kurt}")
            print(f"Correlation: Expected = {self.corr}, Generated = {tree_cor}, Exp-Gen = {self.corr - tree_cor}")'''
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
                logging.info(f"Mean: Expected = {self.mu}, Generated = {tree_mean}, Exp-Gen = {self.mu - tree_mean}")
                logging.info(f"Std Dev: Expected = {self.sigma}, Generated = {tree_std}, Exp-Gen = {self.sigma - tree_std}")
                logging.info(f"Skewness: Expected = {self.skew}, Generated = {tree_skew}, Exp-Gen = {self.skew - tree_skew}")
                #logging.info(f"Kurtosis: Expected = {self.kur}, Generated = {tree_kurt}, Exp-Gen = {self.kur - tree_kurt}")
                logging.info(f"Correlation: Expected = {self.corr}, Generated = {tree_cor}, Exp-Gen = {self.corr - tree_cor}")

        if counter >= 100:
            raise RuntimeError("Good quality arbitrage-free scenarios not found after 100 attempts")
        else:
            end_t = time.time()
            logging.info(f"Computational time to build the tree:{end_t - start_t} seconds")
        
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
        
        return probs, prices # return probababilities and prices to add nodes to the tree
