import numpy as np


def chi_squared(T_model, T_data, variance): # take lists of model prediction, data, and variance of same length
    chi_squared_sum = 0
    for i in range(len(T_model)):
        chi_squared_sum+=(T_model[i]-T_data[i])**2/variance[i]**2
    return chi_squared_sum


# def log_likelihood(p0, m_chi, T_data, var, clusters):
#     #if p0[0]<0 or p0[1]<0:
#     #   return -np.inf
#     T_model = [c.pred_T_b_small_m(p0, m_chi) for c in clusters]
#     X2 = chi_squared(T_model, T_data, var)
#     return (-X2/2)

def log_likelihood_1(p0, T_data, var, clusters, m_chi):
    #if p0[0]<0 or p0[1]<0:
    #   return -np.inf
    T_model = [c.pred_T_b(p0, m_chi) for c in clusters]
    X2 = chi_squared(T_model, T_data, var)
    return (-X2/2)

def log_likelihood(p0, T_data, var, clusters):
    #if p0[0]<0 or p0[1]<0:
    #   return -np.inf
    T_model = [c.pred_T_b(p0) for c in clusters]
    X2 = chi_squared(T_model, T_data, var)
    return (-X2/2)