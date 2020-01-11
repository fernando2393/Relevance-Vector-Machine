import numpy as np

def linear_spline(x_m, x_n):
    compute_kernel = np.zeros((x_m.shape[0], x_n.shape[0]))
    for i in range(x_m.shape[0]):
        for j in range(x_n.shape[0]):
            xmin = np.minimum(x_m[i], x_n[j])
            compute_kernel[i,j] = (1 + x_m[i] * x_n[j] + x_m[i] * x_n[j] * xmin - 
            ((x_m[i] + x_n[j]) / 2) * pow(xmin, 2) + pow(xmin, 3) / 3).prod()

    return compute_kernel

def exponential(x_m, x_n):
    eta_1 = 997*1e-4 
    eta_2 = 2*1e-4
    compute_kernel = np.zeros((x_m.shape[0], x_n.shape[0]))
    for i in range(x_m.shape[0]):
        aux = np.zeros(x_m.shape[1])
        for j in range(x_n.shape[0]):
            xmin_1 = np.minimum(x_m[i,0], x_n[j,0])
            xmin_2 = np.minimum(x_m[i,1], x_n[j,1])
            compute_kernel[i,j] = np.exp(-eta_1 * pow(xmin_1, 2) - eta_2 * pow(xmin_2, 2))
    
    return compute_kernel
