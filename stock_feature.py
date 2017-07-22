# coding=utf-8
import os
import random

import nltk
from nltk.classify import apply_features

import stock_download as sd


def stock_load(file_name):
    """
    从指定数据文件中加载股票数据
    :param file_name: 文件名
    :return: 数据列表[(开盘价, 最高价, 收盘价, 最低价, 价格变动, 涨跌幅)]
    """
    stock_list = []
    file = open(file_name)

    is_first_line = True
    for line in file:
        if is_first_line:
            is_first_line = False
            continue
        prop_list = line.split(',')
        open_price = float(prop_list[1])
        high_price = float(prop_list[2])
        close_price = float(prop_list[3])
        low_price = float(prop_list[4])
        price_change = float(prop_list[6])
        p_change = int(float(prop_list[7]))
        stock_list.append((open_price, high_price, close_price, low_price, price_change, p_change))
    file.close()
    return stock_list


def stock_split(data_list, days=5):
    """
    股票数据分割，将某天涨跌情况和前几天数据关联在一起
    :param data_list: 股票数据列表
    :param days: 关联的天数
    :return: [([day1, day2, ...], label), ...]
    """
    stock_days = []
    for n in range(0, len(data_list) - days):
        before_days = []
        for i in range(1, days + 1):
            before_days.append(data_list[n + i])

        if data_list[n][4] > 0.0:
            label = '+'
        else:
            label = '-'
        stock_days.append((before_days, label))

    return stock_days


def stock_feature(before_days):
    """
    股票特征提取
    :param before_days: 前几日股票数据
    :return: 股票特征
    """
    features = {}
    for n in range(0, len(before_days)):
        stock = before_days[n]
        open_price = stock[0]
        high_price = stock[1]
        close_price = stock[2]
        low_price = stock[3]
        price_change = stock[4]
        p_change = stock[5]
        features['Day(%d)PriceIncrease' % (n + 1)] = (price_change > 0.0)
        features['Day(%d)High==Open' % (n + 1)] = (high_price == open_price)
        features['Day(%d)High==Close' % (n + 1)] = (high_price == close_price)
        features['Day(%d)Low==Open' % (n + 1)] = (low_price == open_price)
        features['Day(%d)Low==Close' % (n + 1)] = (low_price == close_price)
        features['Day(%d)Close>Open' % (n + 1)] = (close_price > open_price)
        features['Day(%d)PChange' % (n + 1)] = p_change

    return features


def train_model(industry_name, os1=None):
    """
    训练模型
    :param industry_name: 行业名称
    """
    stock_industry = sd.stock_industry_load('data/stock_industry.csv')
    electronic_stocks = stock_industry[industry_name]

    stock_days = []
    for stock in electronic_stocks:
        file_name = 'data/%s-%s-%s.csv' % (industry_name, stock[0], stock[1])
        if os.path.exists(file_name):
            stock_data = stock_load(file_name)
            stock_days.append((stock[0], stock_split(stock_data, 5)))

    print(len(stock_days))
    random.shuffle(stock_days)

    print "stock_days:", stock_days[:3]

    train_set_size = int(len(stock_days) * 0.6)
    train_stock = [feature for (code, feature) in stock_days[:train_set_size]]
    test_stock = [feature for (code, feature) in stock_days[train_set_size:]]
    test_index = [code for (code, feature) in stock_days[train_set_size:]]

    train_set = apply_features(stock_feature, train_stock, True)
    test_set = apply_features(stock_feature, test_stock, True)
    test_set1 = apply_features(stock_feature, test_stock, True)
    test_set1 = [feature for (feature, label) in test_set]

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, train_set))
    print(nltk.classify.accuracy(classifier, test_set))

    classifier.show_most_informative_features(20)

    for (code, (test, label)) in stock_days[train_set_size:]:
        # print "test:", test
        print "pro test:", classifier.classify(), " lable:", label



if __name__ == '__main__':
    train_model('电子信息')
