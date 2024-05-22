import numpy as np
from scenario_tree.checkarbitrage import check_arbitrage_prices

# Prima prova: modello binomiale con r=0.05, u=0.15, d=0.08.
act1 = np.array([1, 5])
x_prova1 = np.array([[1.05, 1.05],
                    [5*1.08, 5*1.15]]) 

# Dovrebbe esserci arbitraggio
prova1 = check_arbitrage_prices(act1, x_prova1)
print(prova1)


# Seconda prova: modello binomiale con r=0.05, u=0.03, d=0.01
act2 = np.array([1, 5])
x_prova2 = np.array([[1.05, 1.05],
                    [5*1.01, 5*1.03]]) 

# Dovrebbe esserci arbitraggio
prova2 = check_arbitrage_prices(act2, x_prova2)
print(prova2)


# Terza prova: modello binomiale con r=0.05, u=1.08, d=1.03
act3 = np.array([1, 5])
x_prova3 = np.array([[1.05, 1.05],
                    [5*1.03, 5*1.08]]) 

# Dovrebbe NON esserci arbitraggio
prova3 = check_arbitrage_prices(act3, x_prova3)
print(prova3)