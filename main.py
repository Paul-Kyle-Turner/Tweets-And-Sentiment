import html
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import metrics
from nltk.corpus import stopwords


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


def doc_from_data(data):
    temp_data = []
    for ind in data:
        temp_data.append(ind[2])
    return temp_data


def class_from_data(data):
    temp_data = []
    for ind in data:
        temp_data.append(ind[1])
    return temp_data


if __name__ == '__main__':

    file_path = files_to_csv('Training.txt', 'csv_data.txt')
    data = add_csv_only_three_col(file_path)
    replacer_list(data, full_replace=False, digit_replace=True, handle_replace=True)

    data = data[1::]

    data_class = class_from_data(data)
    data_content = doc_from_data(data)

    del data

    X_train, X_test, y_train, y_test = train_test_split(data_content, data_class, test_size=0.33)

    del data_class
    del data_content

    print("Start of tfidf")

    tfidf_vector = TfidfVectorizer(max_features=5000, max_df=2, stop_words=stopwords.words('english'), dtype='float16')
    tfidf_vector.fit(X_train)
    X_train_tfidf = tfidf_vector.transform(X_train).toarray()
    X_test_tfidf = tfidf_vector.transform(X_test).toarray()

    del X_train
    del X_test

    """count_vector = CountVectorizer()
    X_train_counts = count_vector.fit_transform(X_train)

    tfidf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)

    X_test_counts = count_vector.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)"""

    """print("Start of Classification")

    gnb = GaussianNB()
    gnb.fit(X_train_tfidf, y_train)
    gnb_predicted = gnb.predict(X_test_tfidf)
    print(np.mean(gnb_predicted == y_test))
    del gnb_predicted

    mnb = MultinomialNB()
    mnb.fit(X_train_tfidf, y_train)
    mnb_predicted = mnb.predict(X_test_tfidf)
    print(np.mean(mnb_predicted == y_test))
    del mnb_predicted

    bnb = BernoulliNB()
    bnb.fit(X_train_tfidf, y_train)
    bnb_predicted = bnb.predict(X_test_tfidf)
    print(np.mean(bnb_predicted == y_test))
    del bnb_predicted"""


    """
    with open('newTraining3.txt', 'w+') as file:
        for ind in data:
            file.write(str(ind) + '\n')
    """


    """
    #corpus = data_corpus(data)
    #total_pos, total_neg = pos_neg_totals(data, 1)
    #corpus_p_n = corpus_to_list(corpus)
    #corpus_counts = corpus_pos_neg(data, total_pos, total_neg, corpus_p_n)

    #with open('corpus_counts.p', 'wb') as file:
        #pickle.dump(corpus_counts, file)
    """


    """
    USE THIS PATTERN
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    
    def data_corpus(data):
    corpus = set()
    for ind in data:
        corpus.update(ind[2].split(' '))
    return corpus


def pos_neg_totals(data, y):
    pos = 0
    neg = 0
    for ind in data:
        if ind[y] == '+':
            pos += 1
        else:
            neg += 1
    return pos, neg


def corpus_to_list(corpus):
    corpus = list(corpus)
    for ind in range(len(corpus)):
        corpus[ind] = [corpus[ind], 0, 0, 0, 0]
    return corpus


def corpus_pos_neg(data, pos, neg, corpus):
    print("starting the biggy")
    for ind in data:
        for i in ind[2].split(' '):
            cont = True
            k = 0
            while cont:
                if corpus[k][0] == i:
                    if ind[1] == '+':
                        cont = False
                        corpus[k][1] += 1
                    else:
                        cont = False
                        corpus[k][2] += 1
                k += 1
    for ind in corpus:
        ind[3] = ind[1]/pos
        ind[4] = ind[2]/neg

    return corpus
    """


