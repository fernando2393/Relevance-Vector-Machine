import numpy as np

def linear_spline(x_m, x_n):
    compute_kernel = np.zeros((x_m.shape[0], x_n.shape[0]))
    for i in range(x_m.shape[0]):
        for j in range(x_n.shape[0]):
            xmin = np.minimum(x_m[i], x_n[j])
            compute_kernel[i,j] = (1 + x_m[i] * x_n[j] + x_m[i] * x_n[j] * xmin - 
            ((x_m[i] + x_n[j]) / 2) * pow(xmin, 2) + pow(xmin, 3) / 3).prod()

    return compute_kernel
