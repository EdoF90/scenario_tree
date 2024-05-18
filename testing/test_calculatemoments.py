import numpy as np
from scenario_tree.momentmatching.calculatemoments import mean, std, skewness, kurtosis, correlation

'''
Little example with 2 assets and 3 scenarios to test calculating moments functions
'''
x = np.array([[1, 4, 5],
             [9, 1, 2]])

p = np.array([0.2, 0.5, 0.3])

mu = mean(x, p)
print(mu)

sigma = std(x, p)
print(sigma)

skew = skewness(x, p)
print(skew)

kurt = kurtosis(x, p)
print(kurt)

cor = correlation(x, p)
print(cor)

