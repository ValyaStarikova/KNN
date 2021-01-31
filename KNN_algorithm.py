import pymysql
from Metrics import *
from Load_data import *
from Weight_functions import *
table_name = 'ere'
#----------------------------------------------------Чтение данных из БД------------------------------------------------------------------------
cur = create_cursor()
table_columns_description_tuples = get_data_from_base(cur, "DESCRIBE {0}".format(table_name))
parametrs_and_answer_names_string = get_parametrs_and_answer_names_string(table_columns_description_tuples)
rows = get_data_from_base(cur,"SELECT {0} FROM {1} WHERE sampleCode = 'train'".format(parametrs_and_answer_names_string,table_name) )
precedents = make_list_from_tuple(rows)

metrics_array = {'manhattan_distance': manhattan_distance,
                 'euclidean_distance': euclidean_distance,
                 'normalized_Euclidean_distance':normalized_Euclidean_distance,
                 'chebyshev_metric': chebyshev_metric,
                 'cosine_similarity':cosine_similarity,
                 'chi_square':chi_square,
                 'minkowski_with_pow5':minkowski_with_pow5,
                 'square_euclidean_distance':square_euclidean_distance}


def kNN(cases, precendents, num_neighbors, metric_name, weight_name, weights_of_parameters, normalize_data_fl,lower_bounds_for_parametrs_list,upper_bounds_for_parametrs_list):
    prediction_list = []
    for current_case in cases:
        if normalize_data_fl:
            current_case = normalize_vector(current_case, upper_bounds_for_parametrs_list, lower_bounds_for_parametrs_list)
        prediction_list.append(k_nearest_neighbors(current_case, precendents, num_neighbors, metric_name, weight_name, weights_of_parameters,normalize_data_fl,lower_bounds_for_parametrs_list,upper_bounds_for_parametrs_list))
    return prediction_list

def k_nearest_neighbors(current_case, precendents, num_neighbors, metric_name, weight_name, weights_of_parameters,normalize_data_fl,lower_bounds_for_parametrs_list,upper_bounds_for_parametrs_list):
    distances = []

    for precendent in precendents: # вычисляем расстояние до каждого прецедента в БП
        answer = precendent[-1]
        # нормализация данных
        if normalize_data_fl:
            precedent_param = normalize_vector(precendent, upper_bounds_for_parametrs_list, lower_bounds_for_parametrs_list)
        else:
            precedent_param = precendent
        dist = metrics_array[metric_name](precedent_param, current_case, weights_of_parameters)
        distances.append((answer, dist)) #класс и расстояние
    distances.sort(key=lambda tup: tup[1]) #сортируем по возрастанию

    neighbors = []
    if weight_name != '':
        # взвешенный knn
        weights = make_weights(distances[:num_neighbors],weight_name)
        for i in range(num_neighbors):
                neighbors.append((distances[i][0],weights[i])) #distances[i][0][-1] здесь название класса, получается список из кортежей, где на первом месте класс, на втором вес 
        prediction = voting(neighbors) # классифицируем
    else:
        for i in range(num_neighbors):
                neighbors.append((distances[i][0],1)) #везде одинаковый вес
        # голосование
        output_values = [neib[0] for neib in neighbors]
        prediction = max(set(output_values), key=output_values.count) #екущему случаю присваивается решение наибольшего количества голосов k соседей
    return prediction


def make_weights(dist,weight_name):
    result = np.zeros(len(dist))
    sum = 0.0
    if weight_name == 'reversed_distance':
        for i in range(len(dist)):
            result[i] =reversed_distance(dist[i]) 
            sum += result[i]
    if weight_name == 'geometric_progression':
        for i in range(len(dist)):
            result[i] = geometric_progression(i)
            sum += result[i]
    if weight_name == 'linear_rang':
        for i in range(len(dist)):
            result[i] = linear_rang(i,len(dist))
            sum += result[i]
    return result / sum


def voting(neighbors): #Решение выбирают исходя из объекта с большим суммарным весом среди соседей k
    votes = {} # словарь, где ключ - класс, значение - сумма значений весов
    for neib in neighbors:
        if (votes.get(neib[0]) == None):
            votes[neib[0]] = neib[1]
        else:
            votes[neib[0]] += neib[1]
    lst = list(votes.items())
    lst.sort(key = lambda i: i[1],reverse=True) # сортировка по возрастанию суммы весов
    return lst[0][0]


def normalize_vector(vector, upper_bounds_for_parametrs_list, lower_bounds_for_parametrs_list):
    normalized_vector = []
    for param,max_param,min_param in zip(vector, upper_bounds_for_parametrs_list, lower_bounds_for_parametrs_list):
            normalized_vector.append((param - min_param ) / (max_param- min_param ))
    return normalized_vector



