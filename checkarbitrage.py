# -*- coding: utf-8 -*-
import logging
import gurobipy as grb


def check_arbitrage_prices(new_prices, prev_prices):
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
    
    model.setObjective(0) # Feasibility problem
    
    model.optimize()
    
    if model.status == grb.GRB.Status.OPTIMAL:
        return False # The system has a solution, thus no arbitrage
    else:
        logging.info("Arbitrage opportunity in the solution")
        return True # The system has not a solution, thus arbitrage
