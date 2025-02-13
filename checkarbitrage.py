# -*- coding: utf-8 -*-
import logging
import numpy as np
import gurobipy as grb
from gurobipy import GRB


def check_arbitrage_prices(new_prices, prev_prices, r, dt):
    '''Construct an optimization problem to verify the presence of 
    arbitrage in a single timestep: if this problem is infeasible, 
    then an arbitrage stategy exists.'''

    num_scenarios = new_prices.shape[1] # number of children
    num_assets = new_prices.shape[0]
    
    model = grb.Model()
    model.setParam('OutputFlag', 0)
    
    # Decision variables: probabilities
    pi = model.addVars(num_scenarios, lb = 0, name="pi") 
    
    # If a probability measure under which the previous price equals the expected value of the new prices  
    # exists (i.e. the problem is feasible), the absence of abitrage is guaranteed.
    for k in range(num_assets): 
        model.addConstr(
            grb.quicksum(pi[n] * new_prices[k,n] for n in range(num_scenarios)) == prev_prices[k],
            f"eq1_{k}"
        )

    growth_factor = np.exp(r*dt)

    model.addConstr(growth_factor*grb.quicksum(pi[n] for n in range(num_scenarios)) == 1)
    
    model.setObjective(0) # Feasibility problem
    
    model.optimize()
    
    if model.status == grb.GRB.Status.OPTIMAL:
        return False # The system has a solution, thus no arbitrage
    else:
        logging.info("Arbitrage opportunity in the solution")
        return True # The system has not a solution, thus arbitrage
    

# Alternative method 
def make_arbitrage_free_states(parent_stock_prices, children_stock_prices, n_children):
    '''
    Modify the generated states to get arbitrage-free scenarios.
    In a binomial world, imposing that one state lies above the expected value, and the other below.
    In multinomials, we assure that there is at least one child above, and one child below, the expected value.
    Refer to Villaverde (2003) for a similar method.
    '''
    n_shares = len(parent_stock_prices)
    for j in range(0, n_shares):
    
        M = grb.Model("Get arbitrage-free scenarios")
        s = []
        for i in range(n_children): 
            s.append( M.addVar(lb=0, vtype=GRB.CONTINUOUS, name='s'+str(i+1)) )
        # s0 = parent_node[1]
        s0 = parent_stock_prices[j]
        alpha = 0.01 # this could be the risk-free rate or some historical 'mu', check        
        # s_hat = self.next_states[1,:]
        s_hat = children_stock_prices[j,:]

        argmax, argmin = np.argmax(s_hat), np.argmin(s_hat)
        M.addConstr( (s[argmax] - s0)/s0 - alpha >= 0, name='noarb1')
        M.addConstr( (s[argmin] - s0)/s0 + alpha <= 0, name='noarb2')
        M.setObjective(np.sum((s-s_hat)**2) , GRB.MINIMIZE)
        M.Params.LogToConsole = 0  # ...avoid printing all info with m.optimize()
        M.optimize()
        for i in range(n_children):
            # self.next_states[1,i] = M.getVars()[i].X
            children_stock_prices[j,i] = M.getVars()[i].X
        ## check the change
        # print(s_hat)
        # print(self.next_states[1,:])

        return children_stock_prices
