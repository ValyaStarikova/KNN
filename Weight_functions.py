from random import *

def reversed_distance(dist):
    return 1 / dist[1]

def geometric_progression(rang):
    q = random() #значение q, выбирает эксперт. я не нашла информацию по оптимальному значению q, решила рандомить...
    for i in range(rang):
        q *= q
    return q

def linear_rang(rang,number_of_neighbors):
    return (number_of_neighbors + 1 - rang) / number_of_neighbors

