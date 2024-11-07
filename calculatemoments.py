# -*- coding: utf-8 -*-
import numpy as np

def mean(x, p, matrix = 0):
    mean = np.dot(x, p)
    if matrix == 0:
        return mean
    else:
        mean_matrix = np.tile(np.reshape(mean, (-1, 1)), (1, x.shape[1]))
        return mean_matrix
    
def std(x, p):
    centered_x = x - mean(x, p, matrix=1)
    centered_x_squared = centered_x * centered_x
    variance = np.dot(centered_x_squared, p)
    std_deviation = np.sqrt(variance)
    return std_deviation
    
def skewness(x, p):
    mu = mean(x, p, matrix=1)
    sigma = std(x, p)
    standardized_x = x - mu
    num_azioni = x.shape[0]
    for i in range(num_azioni):
        if (sigma[i]<= 1e-3) and (sigma[i]>= -1e-3):
            standardized_x[i,:] = 0
        else:
            standardized_x[i,:] = standardized_x[i,:] / sigma[i]
        
    standardized_x_cubic = (standardized_x)**3
    skew = np.dot(standardized_x_cubic, p)
    return skew
    
def kurtosis(x, p):
    mu = mean(x, p, matrix=1)
    sigma = std(x, p)
    standardized_x = x - mu
    num_azioni = x.shape[0]
    for i in range(num_azioni):
        if (sigma[i]<= 1e-3) and (sigma[i]>= -1e-3):
            standardized_x[i,:] = 0
        else:
            standardized_x[i,:] = standardized_x[i,:] / sigma[i]
       
    standardized_x_fourth = (standardized_x)**4
    kurt = np.dot(standardized_x_fourth, p)
    return kurt
    
def correlation(x, p):
    mu = mean(x, p, matrix=1)
    sigma = std(x, p)
    centered_x = x - mu
    num_azioni = x.shape[0]
    num_scenarios = x.shape[1]
    cov = np.zeros((num_azioni, num_azioni))
    cor = np.zeros((num_azioni, num_azioni))
    for i in range(num_azioni):
        for j in range(i, num_azioni, 1):
            if i == j:
                cov[i,j] = 1
                cor[i,j] = 1
            elif i != j:
                for s in range(num_scenarios):
                    cov[i,j] += centered_x[i,s] * centered_x[j,s] * p[s]
                cov[j,i] = cov[i,j]
                cor[i,j] = cov[i,j] / (sigma[i]*sigma[j])
                cor[j,i] = cor[i,j]

    return cor


def second_moment(x, p):
    second_moment = np.dot(x**2, p)
    
    return second_moment
