import numpy as np
import pandas as pd
import re
import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle


def grab_file(file_path, mode='w'):
    return open(file=file_path, mode=mode)


def files_to_csv(file_path_in, file_path_out):
    read = grab_file(file_path_in, mode='r+')
    write = grab_file(file_path_out, mode='w+')
    data = read.readlines()
    for x in data:
        write.write(x.replace('\t', ','))
    read.close()
    write.close()
    return file_path_out


def space_seperated_to_data(file_path, cols):
    file = grab_file(file_path, mode='r+')
    list_data = list()
    for x in file.readlines():
        temp = x.split('\t', cols)
        temp[1] = temp[1].replace('\n', '')
        list_data.append(temp)
    file.close()
    return list_data


def html_encoding_to_utf8(string):
    temp = re.search(r'&[a-zA-Z]+;', string)
    if temp is not None:
        last_temp = temp.group()
    i = 0
    while temp is not None and i <= 10:
        find = html.unescape(string[temp.start(): temp.end()])
        if find != '&':
            string = string[:temp.start()] + find + string[temp.end():]
        else:
            string = string[:temp.start()] + ' ' + find + ' ' + string[temp.end():]
        temp = re.search(r'&[a-zA-Z]+;', string)
        if temp is not None:
            if temp.group() == last_temp:
                i += 1
            else:
                last_temp = temp.group()
    return string


def digit_replacer(string):
    temp2 = re.sub(r'(-?\d+)\/(\d+)\s+([+\-*/])\s+(-?\d+)\/(\d+)', ' #number ', string)
    return temp2


def handle_replacer(string):
    temp2 = re.sub(r'@[\S]*', ' @person ', string)
    return temp2


def replacer(string):
    string = string.replace('\n', '')
    temp3 = re.sub(r'https?://\S*|www\S*', '', string)
    temp4 = re.sub(r'!+|[.]+]', '', temp3)
    temp5 = re.sub(r'[?]+', '', temp4)
    temp7 = html_encoding_to_utf8(temp5)
    temp8 = re.sub(r'=', ' ', temp7)
    temp9 = re.sub(r'["]|[(]|[)]|[,]|[-]|[*]', '', temp8)
    temp0 = re.sub(r'\s*&\s*', ' and ', temp9)
    temp10 = re.sub(r'not \s+', 'not-', temp0)
    temp11 = re.sub(r'y{1,1000}', 'y', temp10)
    temp12 = re.sub(r'o{1,1000}', 'o', temp11)
    temp13 = re.sub(r'[\s]{2,1000}', ' ', temp12)
    return temp13


def add_csv_only_three_col(file, cols=2):
    file = grab_file(file, mode='r+')
    list_data = list()
    for x in file.readlines():
        temp = x.split(',', cols)
        temp[2] = temp[2].replace('\n', '')
        list_data.append(temp)
    file.close()
    return list_data


def replacer_list(data_set, full_replace=False, digit_replace=False, handle_replace=False):
    if digit_replace:
        for data in data_set:
            data[2] = digit_replacer(data[2].lower())
    if handle_replace:
        for data in data_set:
            data[2] = handle_replacer(data[2].lower())
    if full_replace:
        for data in data_set:
            data[2] = replacer(data[2].lower())


if __name__ == '__main__':

    full_ber_bayes = pickle.load(open('ber_bayes_full_replace.p', 'rb'))
    #full_replace_bayes = pickle.load(open('full_replace_bayes.p', 'rb'))
    #handle_number_bayes = pickle.load(open('handle_number_bayes.p', 'rb'))
    handle_bayes = pickle.load(open('ber_bayes_handle_replace.p', 'rb'))
    no_replace_bayes = pickle.load(open('ber_bayes_no_replace.p', 'rb'))

    bayes = [full_ber_bayes, handle_bayes, no_replace_bayes]

    #for bae in bayes:
    #    print(bae)
    #    print(bae.best_score_)

    data1 = space_seperated_to_data('test1_public.txt', 1)[1::]
    data2 = space_seperated_to_data('test2_public.txt', 1)[1::]

    dataframe1 = pd.DataFrame(data1, columns=['id', 'content'])
    dataframe2 = pd.DataFrame(data2, columns=['id', 'content'])

    tfidf_full_replace = pickle.load(open('tfidf_ber_bayes_full_replace.p', 'rb'))
    tfidf_handle_replace = pickle.load(open('tfidf_ber_bayes_handle_replace.p', 'rb'))
    tfidf_no_replace = pickle.load(open('tfidf_ber_bayes_no_replace.p', 'rb'))

    tfidf_replace = [tfidf_full_replace, tfidf_handle_replace, tfidf_no_replace]

    names = ['ber_bayes_full_replace', 'ber_bayes_handle_replace', 'ber_bayes_no_replace']

    for i in range(3):
        data_temp1 = tfidf_replace[i].transform(dataframe1['content'])
        data_temp2 = tfidf_replace[i].transform(dataframe2['content'])

        pred1 = bayes[i].predict(data_temp1)
        pred2 = bayes[i].predict(data_temp2)

        file = open(names[i] + '_output.txt', 'w+')
        for j in range(len(pred1)):
            ident = dataframe1.iloc[j][0]
            file.write(f'{ident} {pred1[j]}\n')

        for j in range(len(pred2)):
            ident = dataframe2.iloc[j][0]
            file.write(f'{ident} {pred2[j]}\n')
        file.close()








