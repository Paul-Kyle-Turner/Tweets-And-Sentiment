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


def grab_file(file_path, mode='w', encoding='utf8'):
    return open(file=file_path, mode=mode, encoding=encoding)


def files_to_csv(file_path_in, file_path_out):
    read = grab_file(file_path_in, mode='r+')
    write = grab_file(file_path_out, mode='w+')
    data = read.readlines()
    for x in data:
        write.write(x.replace('\t', ','))
    read.close()
    write.close()
    return file_path_out


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

# honestly testing git
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
    file_path = files_to_csv('Training.txt', 'csv_data.txt')
    data = add_csv_only_three_col(file_path)
    replacer_list(data, full_replace=False, digit_replace=False, handle_replace=False)

    data = data[1::]

    df = pd.DataFrame(data, columns=['id', 'sentiment', 'text'])
    del data

    xtrain, xtest, ytrain, ytest = train_test_split(df['text'], df['sentiment'], test_size=0.1, shuffle=True)

    tfidf_vector = TfidfVectorizer(stop_words='english')
    tfidf_vector.fit(xtrain)

    pickle.dump(tfidf_vector, open('tfidf_ber_bayes_no_replace.p', 'wb'))

    nb = BernoulliNB()
    #rf = RandomForestClassifier(n_estimators=1000)
    xtrain1 = tfidf_vector.transform(xtrain)
    nb.fit(xtrain1, ytrain)
    pred1 = nb.predict(tfidf_vector.transform(xtest))

    print(accuracy_score(ytest, pred1))
    print(confusion_matrix(ytest, pred1))

    gcv = GridSearchCV(nb, {'alpha': [1.5, 2, 3, 4, 10, 100, 1.0, 0.1, 0.001, 0.0001], 'fit_prior': [True, False]})
    #gcv = GridSearchCV(rf, {'n_estimators': [100, 1000], 'criterion': ['gini', 'entropy']}, n_jobs=3)
    #gcv = GridSearchCV(svc, {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'decision_function_shape ': ['ovo', 'ovr']})
    gcv.fit(xtrain1, ytrain)

    print(gcv.best_score_, gcv.best_params_)

    with open('ber_bayes_no_replace.p', 'wb') as file:
        pickle.dump(gcv, file)

