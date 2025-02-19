import numpy as np
import random 

def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices
    
def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):

    current_theta_0 = np.array(current_theta_0)

    if label*(np.dot(current_theta, feature_vector) + current_theta_0) <= 0:
        current_theta += label*feature_vector
        current_theta_0 += label

    return current_theta, current_theta_0 
    
def perceptron(feature_matrix, labels, T):
    (nsamples, nfeatures) = feature_matrix.shape
    theta = np.zeros(nfeatures)
    theta_0 = 0.0
    for t in range(T):
        for i in get_order(nsamples):
            theta, theta_0 = perceptron_single_step_update(
                feature_matrix[i], labels[i], theta, theta_0)
    return (theta, theta_0)


def average_perceptron(feature_matrix, labels, T):
    (nsamples, nfeatures) = feature_matrix.shape
    theta     = np.zeros(nfeatures)
    theta_sum = np.zeros(nfeatures)
    theta_0 = 0.0
    theta_0_sum = 0.0
    for _ in range(T):
        for i in get_order(nsamples):
            theta, theta_0 = perceptron_single_step_update(
                feature_matrix[i], labels[i], theta, theta_0)
            theta_sum += theta
            theta_0_sum += theta_0
    return (theta_sum / (nsamples * T), theta_0_sum / (nsamples * T))

