import numpy as np
from checkarbitrage import check_arbitrage_prices

# Prima prova: modello binomiale con r=0.05, u=0.15, d=0.08.
# act1 = np.array([1, 5])
x_prova1 = np.array([[0.05, 0.05],
                    [0.08, 0.15]]) 

# Dovrebbe esserci arbitraggio
prova1 = check_arbitrage_prices(x_prova1)
print(prova1)


# Seconda prova: modello binomiale con r=0.05, u=0.03, d=0.01
# act2 = np.array([1, 5])
x_prova2 = np.array([[0.05, 0.05],
                    [0.01, 0.03]]) 

# Dovrebbe esserci arbitraggio
prova2 = check_arbitrage_prices(x_prova2)
print(prova2)


# Terza prova: modello binomiale con r=0.05, u=1.08, d=1.03
#act3 = np.array([1, 5])
x_prova3 = np.array([[0.05, 0.05],
                    [0.03, 0.08]]) 

# Dovrebbe NON esserci arbitraggio
prova3 = check_arbitrage_prices(x_prova3)
print(prova3)


x_prova4 = np.array([[0.03, -0.12, 0.14],
                     [0.05, -0.02, 0.25]])
prova4 = check_arbitrage_prices(x_prova4)
print(prova4)

x_prova5 = np.array([[0.03, -0.12, 0.14],
                     [0.05, -0.25, 0.25]])
prova5 = check_arbitrage_prices(x_prova5)
print(prova5)