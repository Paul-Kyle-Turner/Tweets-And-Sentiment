import pandas as pd
import numpy as np
import html
import re

# GIT TEST


def grab_file(file_path, mode='w', encoding='utf8'):
    return open(file=file_path, mode=mode, encoding=encoding)


def files_to_csv(file_path):
    read = grab_file(file_path, mode='r+')
    file_path = 'new' + file_path
    write = grab_file(file_path, mode='w+')
    data = read.readlines()
    for x in data:
        write.write(x.replace('\t', ','))
    read.close()
    write.close()
    return file_path


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


def replacer(string):
    string = string.replace('\n','')
    temp2 = re.sub(r'@[\S]*', '@person', string)
    temp3 = re.sub(r'https?://\S*', '', temp2)
    temp4 = re.sub(r'!+', '!', temp3)
    temp5 = re.sub(r'[?]+', '?', temp4)
    temp6 = html_encoding_to_utf8(temp5)
    temp7 = re.sub(r'\s+', ' ', temp6)
    return temp7


def add_csv_only_three_col(file):
    file = grab_file(file, mode='r+')
    first_line = file.readline().split(sep=',')
    list_data = list()
    for x in file.readlines():
        temp = x.split(',', 2)
        temp[2] = temp[2].replace('\n', '')
        list_data.append(temp)
    return list_data


def replacer_list(data_set):
    for data in data_set:
        data[2] = replacer(data[2])


if __name__ == '__main__':
    file_path = files_to_csv('Training.txt')
    file_path = 'newTraining.txt'
    data = add_csv_only_three_col(file_path)
    replacer_list(data)
    frame = pd.DataFrame(data[1:], columns=[data[0][0], data[0][1], data[0][2]])

    frame.to_pickle('training_pickle.p')

