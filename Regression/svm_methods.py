import numpy as np

def linear_spline(x_m, x_n):
    x_m_aux = abs(x_m)
    x_n_aux = abs(x_n)
    compute_kernel = np.zeros((x_m_aux.shape[0], x_n_aux.shape[0]))
    for i in range(x_m_aux.shape[0]):
        for j in range(x_n_aux.shape[0]):
            xmin = np.minimum(x_m_aux[i], x_n_aux[j])
            compute_kernel[i,j] = (1 + x_m_aux[i] * x_n_aux[j] + x_m_aux[i] * x_n_aux[j] * xmin - 
            ((x_m_aux[i] + x_n_aux[j]) / 2) * pow(xmin, 2) + pow(xmin, 3) / 3).prod()

    return compute_kernel