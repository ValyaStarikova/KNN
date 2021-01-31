import pymysql
import numpy as np

def make_string_from_list(splitted_list):
    string_with_comma_separated_elements = ', '.join(str(element) for element in splitted_list)
    
    return string_with_comma_separated_elements
#формируем строку имен параметров и решения прецедентов
def get_parametrs_and_answer_names_string(table_columns_description_tuples):
    parametrs_and_answer_names_list = [str[0] for str in table_columns_description_tuples if (str[0] != 'id') and (str[0] != 'sampleCode') and (str[0] != 'qualityCode')] #['Col0', 'Col1', 'Col2', 'Col3', 'Col4']
    parametrs_and_answer_names_string = make_string_from_list(parametrs_and_answer_names_list)

    return parametrs_and_answer_names_string

def make_list_from_tuple(splitted_tuple):
    result_list = [elem for elem in splitted_tuple]
    
    return result_list

def get_data_from_base(cur, query):
    cur.execute(query)#Из объекта подключения con создается курсор. Курсор используется для перемещения записей из набора результатов.Для использования команды SQL вызывается метод курсора execute()
    data_from_base = cur.fetchall()#Метод fetchall() позволяет получить все записи. Он возвращает набор результатов. Технически, это кортеж из кортежей. Каждый из внутренних кортежей представляет собой строку в таблице.
    
    return data_from_base
      
def  create_cursor():
    con = pymysql.connect(host='localhost',
        user='root',
        password='valyastar111A',
        db='table_storage')
    return con.cursor()

def make_list_of_lists_from_tuple_of_tuples(splitted_tuple):
    arr = [np.array(row) for row in splitted_tuple]    
    return np.array(arr)
