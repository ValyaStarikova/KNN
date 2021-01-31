import pymysql
from Metrics import *
from Load_data import *
from Weight_functions import *
from KNN_algorithm import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

table_name = 'ere'
#----------------------------------------------------Чтение данных из БД------------------------------------------------------------------------
cur = create_cursor()
table_columns_description_tuples = get_data_from_base(cur, "DESCRIBE {0}".format(table_name))
parametrs_and_answer_names_string = get_parametrs_and_answer_names_string(table_columns_description_tuples)
precedents = make_list_from_tuple(get_data_from_base(cur,"SELECT {0} FROM {1} WHERE sampleCode = 'train'".format(parametrs_and_answer_names_string,table_name)))
cases = make_list_from_tuple(get_data_from_base(cur,"SELECT {0} FROM {1} WHERE sampleCode = 'test'".format(parametrs_and_answer_names_string,table_name)))
parametrs_description_tuples = get_data_from_base(cur, "SELECT * FROM {0}".format(table_name + 'Description'))

print(len(cases))

#списки нижних и верхних границ диапазонов параметров 
lower_bounds_for_parametrs_list = [str[5] for str in parametrs_description_tuples]
upper_bounds_for_parametrs_list = [str[6] for str in parametrs_description_tuples]
#print(upper_bounds_for_parametrs_list)
#print(lower_bounds_for_parametrs_list)
#-----------------------------------------------MODE SETTINGS----------------------------------------------------
weight_param_fl = False # Взвешивать ли параметры
normalize_data_fl = False # Нормализовывать данные или нет
weight_function = '' # reversed_distance # geometric_progression # linear_rang Функции весов
metric_name = 'euclidean_distance' #euclidean_distance # normalized_Euclidean_distance # chebyshev_metric # cosine_similarity # chi_square # minkowski_with_pow5
num_neighbors = 3
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
#составляем meshgrid
#подаем meshgrid в KNN рисуем по ней карту 
# сверху выводим карту точек
clss = {'Iris-versicolor':0,
		'Iris-setosa':1,
		'Iris-virginica':2}
'''pad = 1
x_min, x_max = lower_bounds_for_parametrs_list[1] - pad,upperdata:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYIAAAEGCAYAAABo25JHAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAAyhUlEQVR4nO3de3xU9bno/8+TmVAYkHIUPF4wM7CPrRsSQAzxAloR6qVS7UVf1EaL7kpKUn/28nMf3b/8DtuXp+nePdrLsd3Bpu222IzKT7q17v701IKioq0KCCIqipggyq6IR24BS8Jz/phJTCYzk5mVWWvWrHnevuaVzHfW5btmZJ6s9XzX8xVVxRhjTPmqKHYHjDHGFJcFAmOMKXMWCIwxpsxZIDDGmDJngcAYY8pcuNgdyNf48eM1FosVuxvGGFNS1q9f/76qTkj3WskFglgsxrp164rdDWOMKSki0pnpNVcvDYnId0Rki4i8LCL3icjIlNdFRO4UkW0i8pKIzHSzP8YYYwZzLRCIyMnAjUCtqlYDIeArKYtdApyafDQAy9zqjzHGmPTcThaHgVEiEgYiwLspr18O3KMJfwbGiciJLvfJGGNMP67lCFT1HRG5A9gBHAIeU9XHUhY7GXi73/OdybZd/RcSkQYSZwxUVVUN2teRI0fYuXMnhw8fLtwBmGEbOXIkEydOpLKysthdMcZk4VogEJH/ROIv/knAh8ADInK1qrb3XyzNqoOKH6lqG9AGUFtbO+j1nTt3cswxxxCLxRBJt0njNVVlz5497Ny5k0mTJhW7O8aYLNy8NDQfeEtVd6vqEeDfgHNSltkJnNLv+UQGXz4a0uHDhznuuOMsCPiIiHDcccfZWZopiDhxYsSooIIYMeLEi92lQHEzEOwAzhKRiCS+oecBr6Ys8zDwteToobOAvaq6K3VDubAg4D/2mZhCiBOngQY66URROumkgQYLBgXkWiBQ1eeAlcAGYHNyX20iskREliQXewTYDmwDfgE0udUfY0xpaqaZLroGtHXRRTPNRepR8Lg6akhV/1FVT1PValW9RlU/UtW7VPWu5Ouqqt9U1b9R1RpVLdk7xcaMGZPxtXPOSb0iVjjf//73Xdu2MX6wgx15tZv8Wa0hF/X09ADw7LPPurYPCwQm6KoYPFIwW7vJX1kGgnj8FWKxNioq7iAWayMef6Vg216zZg1z587lq1/9KjU1NcDHZwu7du3ivPPOY8aMGVRXV/P0008PWn/Lli3U1dUxY8YMpk2bxhtvvAFAe3t7X/s3vvENenp6uOWWWzh06BAzZsygvr4egB/96EdUV1dTXV3NT37yEwAOHjzIpZdeyvTp06murmbFihUA3HbbbcyaNYvq6moaGhqw2eqMH7XQQoTIgLYIEVpoKVKPAkhVS+pxxhlnaKpXXnllUFsm7e1bNBL5scLtfY9I5Mfa3r4l522kM3r0aFVVfeKJJzQSiej27dsHvXbHHXfo9773PVVV7e7u1n379g3azg033KDt7e2qqvrRRx9pV1eXvvLKK7pgwQL961//qqqqjY2Nunz58gHbVlVdt26dVldX64EDB3T//v06ZcoU3bBhg65cuVKvv/76vuU+/PBDVVXds2dPX9vVV1+tDz/88LDeg3Ty+WyMyaRd2zWqURUVjWpU27W92F0qOcA6zfC9WnZnBM3Na+nq6h7Q1tXVTXPz2oLto66uLu3Y+VmzZnH33Xdz6623snnzZo455phBy5x99tl8//vf5wc/+AGdnZ2MGjWK1atXs379embNmsWMGTNYvXo127dvH7Tu2rVr+eIXv8jo0aMZM2YMX/rSl3j66aepqalh1apV3HzzzTz99NN88pOfBOCJJ57gzDPPpKamhscff5wtW7YU7D0wppDqqaeDDo5ylA46qKe+2F0KlLILBDt27Mur3YnRo0enbT/vvPN46qmnOPnkk7nmmmu45557ePDBB5kxYwYzZsxg3bp1fPWrX+Xhhx9m1KhRXHTRRTz++OOoKosWLWLjxo1s3LiRrVu3cuuttw7avma4tPOpT32K9evXU1NTwz/8wz9w2223cfjwYZqamli5ciWbN29m8eLFNubfmDJVdoGgqmpsXu2F1NnZyfHHH8/ixYv5+te/zoYNG/jiF7/Y9wVfW1vL9u3bmTx5MjfeeCOXXXYZL730EvPmzWPlypW89957AHzwwQd0diYqylZWVnLkyBEgEWgeeughurq6OHjwIA8++CDnnnsu7777LpFIhKuvvpqbbrqJDRs29H3pjx8/ngMHDrBy5UrXj98Y408lNx/BcLW0zKGh4bEBl4cikTAtLXNc3/eaNWu4/fbbqaysZMyYMdxzzz2DllmxYgXt7e1UVlZywgknsHTpUo499li+973vceGFF3L06FEq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_bounds_for_parametrs_list[1] + pad
y_min, y_max = lower_bounds_for_parametrs_list[3] - pad,upper_bounds_for_parametrs_list[3] + pad
h = 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
precedents = make_list_from_tuple(get_data_from_base(cur,"SELECT {0} FROM {1} WHERE sampleCode = 'train'".format('two, col3, col4',table_name)))
predictions = kNN(zip(xx.ravel(),yy.ravel()), precedents, num_neighbors, metric_name, weight_function, weights_of_parameters,normalize_data_fl,lower_bounds_for_parametrs_list,upper_bounds_for_parametrs_list)
prediction_s = [clss[pred] for pred in predictions]
prediction_s = np.array(prediction_s).reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
f = plt.figure(figsize=(10, 10))

plt.pcolormesh(xx, yy, prediction_s, cmap=cmap_light)


precedents_labels =np.array( get_answer(precedents))
cases_labels = np.array( get_answer(cases))
precedents = make_list_from_tuple(get_data_from_base(cur,"SELECT {0} FROM {1} WHERE sampleCode = 'train'".format('two, col3',table_name)))
cases = make_list_from_tuple(get_data_from_base(cur,"SELECT {0} FROM {1} WHERE sampleCode = 'test'".format('two, col3',table_name)))

#выводим график точек где крестик это тестовые данные, а о обучающие 

colours = ['red', 'green', 'blue']
legend = ['Setosa', 'Versicolour', 'Virginica']
classes = list(set(precedents_labels))




for i in classes:
    idx = np.where(precedents_labels == i)
    points = np.array([precedents[j] for j in idx[0]])

    plt.scatter(points[:,0], 
                points[:,1], 
                c=colours[clss[i]], 
                label=i + '_trn')
for i in classes:
    idx = np.where(cases_labels == i)
    points = np.array([cases[j] for j in idx[0]])

    plt.scatter(points[:,0], 
                points[:,1], 
                c=colours[clss[i]], 
                label=i + '_test',
                marker='x')
plt.legend()
plt.title('Iris Dataset', fontsize=16)

plt.show()'''

metrics_array = {'manhattan_distance': manhattan_distance,
                 'euclidean_distance': euclidean_distance,
                 'normalized_Euclidean_distance':normalized_Euclidean_distance,
                 'chebyshev_metric': chebyshev_metric,
                 'cosine_similarity':cosine_similarity,
                 'chi_square':chi_square,
                 'minkowski_with_pow5':minkowski_with_pow5,
                 'square_euclidean_distance':square_euclidean_distance}

acuuracy = []
print(type(metrics_array.keys()))
for metric in metrics_array:
	predictions = kNN(cases, precedents, num_neighbors, metric, weight_function, weights_of_parameters,normalize_data_fl,lower_bounds_for_parametrs_list,upper_bounds_for_parametrs_list)
	acuuracy.append(accuracy_check(answer_of_test_cases, predictions))
f = plt.figure(figsize=(10, 10))

x = range(len(acuuracy))
color = ['lightblue','darkblue','aqua','royalblue','mediumblue','cornflowerblue','steelblue','skyblue']
plt.bar(x, acuuracy, color = color, align='edge')
plt.annotate(accuracy)
plt.xticks(x, metrics_array.keys(),rotation=40,horizontalalignment='right', fontsize=8)
plt.ylabel('Accuracy')
f.tight_layout()
plt.show()

