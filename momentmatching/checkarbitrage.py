import numpy as np
import gurobipy as grb

# Per ora immaginiamo che la funzione prenda in input i prezzi degli asset al tempo t (vettore di lunghezza pari al numero di asset)
# e gli scenari dei prezzi al tempo t+1 (matrice num_asset * num_scenarios)
def check_arbitrage_prices(actualprices, futurescenarios):
    num_scenarios = futurescenarios.shape[1]
    num_assets = futurescenarios.shape[0]
    
    model = grb.Model()
    
    pi = model.addVars(num_scenarios, lb = 0, name="pi")
    
    
    for k in range(num_assets):
        model.addConstr(
            grb.quicksum(pi[n] * futurescenarios[k,n] for n in range(num_scenarios)) == actualprices[k],
            f"eq1_{k}"
        )
    
    model.setObjective(0)
    
    model.optimize()
    
    # Verifica se il problema è risolvibile
    if model.status == grb.GRB.Status.OPTIMAL:
        return False  # Il sistema ha una soluzione, quindi non c'è un arbitraggio
    else:
        return True  # Il sistema non ha una soluzione, quindi c'è arbitraggio
