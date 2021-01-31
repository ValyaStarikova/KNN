import numpy as np
from math import sqrt , pow

def euclidean_distance(current_case, precedent,weights_of_parameters):
    distance = 0.0
    
    for i_param_current_case,i_param_precedent,weight_of_param in zip(precedent, current_case,weights_of_parameters):
        distance += pow(weight_of_param*(i_param_current_case - i_param_precedent),2)
    return sqrt(distance)


def square_euclidean_distance(current_case, precedent,weights_of_parameters):
    distance = 0.0
    
    for i_param_current_case,i_param_precedent,weight_of_param in zip(precedent, current_case,weights_of_parameters):
        distance += pow(weight_of_param*(i_param_current_case - i_param_precedent),2)
    return distance
    

def manhattan_distance(precedent, current_case,weights_of_parameters):
    result_distance = 0.0

    for i_param_precedent, i_param_current_case,weight_of_param in zip(precedent, current_case,weights_of_parameters):
        result_distance += weight_of_param * abs(i_param_precedent - i_param_current_case)

    return result_distance


def chebyshev_metric(precedent, current_case,weights_of_parameters):
    result_distance, current_distance = 0.0, 0.0

    for i_param_precedent, i_param_current_case,weight_of_param in zip(precedent, current_case,weights_of_parameters):
        current_distance = abs(weight_of_param*(i_param_precedent - i_param_current_case))
        
        if current_distance > result_distance:
            result_distance = current_distance

    return result_distance


def normalized_Euclidean_distance(precedent, current_case, weights_of_parameters):
    norm_data_1,norm_data_2,n_dist = 0,0,0

    for i_param_precedent, i_param_current_case, weight_of_param in zip(precedent, current_case,weights_of_parameters):
        norm_data_1 += pow(i_param_precedent * weight_of_param, 2)
        norm_data_2 += pow(i_param_current_case * weight_of_param, 2)

    for i_param_precedent, i_param_current_case, weight_of_param in zip(precedent, current_case,weights_of_parameters):
        n_dist += abs(weight_of_param*(i_param_precedent/norm_data_1 - i_param_current_case /norm_data_2))

    return np.sqrt(n_dist)


def cosine_similarity(precedent, current_case, weights_of_parameters):
    mult, norm_a, norm_b = 0,0,0
    for i_param_precedent, i_param_current_case, weight_of_param in zip(precedent, current_case,weights_of_parameters):
        mult += i_param_precedent * i_param_current_case * weight_of_param
        norm_a += pow(i_param_precedent * weight_of_param, 2)
        norm_b += pow(i_param_current_case * weight_of_param, 2)

    return (1-mult/(sqrt(norm_a)*sqrt(norm_b)))


def chi_square (precedent, current_case,weights_of_parameters):
    result_distance = 0.0

    for i_param_precedent, i_param_current_case,weight_of_param in zip(precedent, current_case,weights_of_parameters):
        result_distance += abs(weight_of_param * (i_param_precedent - i_param_current_case)) / abs(weight_of_param * (i_param_precedent + i_param_current_case))

    return np.sqrt(result_distance)

def minkowski_with_pow5(precedent, current_case, weights_of_parameters):
    result_distance = 0

    for i_param_precedent, i_param_current_case,weight_of_param in zip(precedent, current_case,weights_of_parameters):
        result_distance += pow(weight_of_param * abs(i_param_precedent - i_param_current_case),5)

    return pow(result_distance, 1/5)

