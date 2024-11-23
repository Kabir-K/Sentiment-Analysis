from perceptron import get_order
import numpy as np


def pegasos_single_step_update(feature_vector,label,L,eta,current_theta,current_theta_0):

    if label*(np.dot(feature_vector, current_theta) + current_theta_0) <= 1:
        current_theta = (1 - eta*L)*current_theta + eta*label*feature_vector
        current_theta_0 = current_theta_0 + eta*label 
    else:
        current_theta = (1 - eta*L)*current_theta
    
    return current_theta, current_theta_0



def pegasos(feature_matrix, labels, T, L):

    (nsamples, nfeatures) = feature_matrix.shape
    theta = np.zeros(nfeatures)
    theta_0 = 0
    count = 0
    for t in range(T):
        for i in get_order(nsamples):
            count += 1
            eta = 1 / np.sqrt(count)
            (theta, theta_0) = pegasos_single_step_update(
                feature_matrix[i], labels[i], L, eta, theta, theta_0)
    return (theta, theta_0)