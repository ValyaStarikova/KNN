import pymysql
from Metrics import *
from Load_data import *
from Weight_functions import *
from KNN_algorithm import *

table_name = 'ere'
#----------------------------------------------------Чтение данных из БД------------------------------------------------------------------------
cur = create_cursor()
table_columns_description_tuples = get_data_from_base(cur, "DESCRIBE {0}".format(table_name))
parametrs_and_answer_names_string = get_parametrs_and_answer_names_string(table_columns_description_tuples)
precedents = make_list_from_tuple(get_data_from_base(cur,"SELECT {0} FROM {1} WHERE sampleCode = 'train'".format(parametrs_and_answer_names_string,table_name)))
cases = make_list_from_tuple(get_data_from_base(cur,"SELECT {0} FROM {1} WHERE sampleCode = 'test'".format(parametrs_and_answer_names_string,table_name)))
parametrs_description_tuples = get_data_from_base(cur, "SELECT * FROM {0}".format(table_name + 'Description'))
#списки нижних и верхних границ диапазонов параметров 
lower_bounds_for_parametrs_list = [str[5] for str in parametrs_description_tuples]
upper_bounds_for_parametrs_list = [str[6] for str in parametrs_description_tuples]
#print(upper_bounds_for_parametrs_list)
#print(lower_bounds_for_parametrs_list)
#-----------------------------------------------MODE SETTINGS----------------------------------------------------
weight_param_fl = True # Взвешивать ли параметры
normalize_data_fl = False # Нормализовывать данные или нет
weight_function = 'linear_rang' # reversed_distance # geometric_progression # linear_rang Функции весов
metric_name = 'square_euclidean_distance' #euclidean_distance # normalized_Euclidean_distance # chebyshev_metric # cosine_similarity # chi_square # minkowski_with_pow5
num_neighbors = 6
matrix_of_cases = []

if weight_param_fl:
    weights_of_parameters = [str[7] for str in parametrs_description_tuples]
else:
	weights_of_parameters = [1 for str in parametrs_description_tuples]

predictions = kNN(cases, precedents, num_neighbors, metric_name, weight_function, weights_of_parameters,normalize_data_fl,lower_bounds_for_parametrs_list,upper_bounds_for_parametrs_list)

def get_answer(cases): # cоздает список ответов на тестовую выборку
	answer_list = []
	for case in cases:
		answer_list.append(case[-1])
	return answer_list
answer_of_test_cases = get_answer(cases)

def accuracy_check(answer_list, predictions_list): # проверка точности
	count = 0
	for answer,prediction in zip(answer_list,  predictions_list):
		if answer == prediction:
			count += 1
	return count/len(answer_list)

accuracy = accuracy_check(answer_of_test_cases, predictions)
print('accuracy',accuracy)