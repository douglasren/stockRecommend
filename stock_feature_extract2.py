# coding=utf-8
import os
import sys
import random

import datetime
import nltk
from nltk.classify import apply_features

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
from skopt import gp_minimize

import numpy as np
import pandas as pd

import stock_download as sd
import base_utility as bu

plt.style.use("ggplot")


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

        temp = [float(item) for item in prop_list[1:]]

        # open_price = float(prop_list[1])
        # high_price = float(prop_list[2])
        # close_price = float(prop_list[3])
        # low_price = float(prop_list[4])
        # price_change = float(prop_list[6])
        # p_change = int(float(prop_list[7]))
        # stock_list.append((open_price, high_price, close_price, low_price, price_change, p_change))
        stock_list.append(temp)

    file.close()
    return stock_list


def stock_load_active(file_name='./data/stock_open.csv'):
    """
    从指定数据文件中加载股票数据
    :param file_name: 文件名
    :return: 数据列表[(开盘价, 最高价, 收盘价, 最低价, 价格变动, 涨跌幅)]
    """
    stock_open_list = []
    file = open(file_name)

    is_first_line = True
    for line in file:
        if is_first_line:
            is_first_line = False
            continue
        prop_list = line.split(',')

        stock_open_list.append(prop_list[1])

    file.close()
    return set(stock_open_list)


def is_care_day_before(file_name, before_day):
    """
    判断最新的一天是否在阈值之前
    :param file_name:
    :param before_day:
    :return:
    """

    try:
        with open(file_name) as fp:
            data_list = fp.readlines(5)
            least_date = data_list[1].split(",")[0].strip()

            return least_date <= before_day
    except Exception,ex:
        print "exception:", ex.message
        return True



def stock_load_active_from_data(stocks_list, day_len = 5):
    """
    从指定数据文件中加载股票数据
    :param file_name: 文件名
    :return: 数据列表[(开盘价, 最高价, 收盘价, 最低价, 价格变动, 涨跌幅)]
    """
    open_stocks = []
    before_day = bu.get_before_day(day_len)

    for stock in stocks_list:
        file_name = 'data/%s-%s-%s.csv' % (stock[2], stock[0], stock[1])
        if os.path.exists(file_name):
            if not is_care_day_before(file_name, before_day):
                open_stocks.append(stock[0])

    print "active stock len:", len(open_stocks)
    return set(open_stocks)



def get_stock_info(stock_industry):
    """
    {'行业名':[('股票代码', '股票名', '行业名'), ...]}
    :param stock_industry:
    :return:
    """
    stock_info_map = {}
    for (key, stocks) in stock_industry.items():
        for stock in stocks:
            stock_info_map[stock[0]] = stock

    return stock_info_map


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


def stock_history_feature(data_list, days=5, index=0):
    """
    股票数据分割，将某天涨跌情况和前几天数据关联在一起
    :param data_list: 股票数据列表
    :param days: 关联的天数
    :return: [([day1, day2, ...], label), ...]
    """

    def get_stock_label(n_index):
        try:
            if data_list[n_index][5] > 0.0:
                return 1.0
            else:
                return 0.0
        except Exception, ex:
            print "exception:", ex.message
            return -1.0

    before_days = []
    for start in range(index, min(len(data_list) - days, index + days)):
        before_days.append(data_list[start])

    label = get_stock_label(index - 1)

    return before_days, label


def stock_mutil_history_feature(data_list, days=5, index=0):
    """
    股票数据分割，将某天涨跌情况和前几天数据关联在一起
    :param data_list: 股票数据列表
    :param days: 关联的天数
    :return: [([day1, day2, ...], label), ...]
    """

    def get_stock_label(n_index):
        try:
            if data_list[n_index][5] > 0.0:
                return 1.0
            else:
                return 0.0
        except Exception, ex:
            print "exception:", ex.message
            return -1.0

    res = []
    for i in range(index, len(data_list) - days):
        before_days = []
        for start in range(index, min(len(data_list) - days, index + days)):
            before_days.append(data_list[start])

        label = get_stock_label(index - 1)

        res.append((before_days, label))

    # for i in range(index, len(data_list) - days):
    #     label = get_stock_label(i - 1)
    #     res.append((data_list[i], label))

    return res


def stock_feature(before_days):
    """
    股票特征提取

    0    open：开盘价
    1    high：最高价
    2    close：收盘价
    3    low：最低价
    4    volume：成交量
    5    price_change：价格变动
    6    p_change：涨跌幅
    7    ma5：5日均价
    8    ma10：10日均价
    9    ma20:20日均价
    10   v_ma5:5日均量
    11   v_ma10:10日均量
    12   v_ma20:20日均量
    13   turnover:换手率[注：
    :param before_days: 前几日股票数据
    :return: 股票特征
    """
    features = []
    for i in range(0, len(before_days)):
        stock = before_days[i]
        # print "stock:", stock
        open_price = stock[0]
        high_price = stock[1]
        close_price = stock[2]
        low_price = stock[3]
        volume = stock[4]
        price_change = stock[5]
        p_change = stock[6]
        ma5 = stock[7]
        ma10 = stock[8]
        ma20 = stock[9]
        v_ma5 = stock[10]
        v_ma10 = stock[11]
        v_ma20 = stock[12]
        turnover = stock[13]

        on_feature = [
            price_change,
            high_price - close_price,
            open_price - close_price,
            low_price - close_price,
            ma5,
            ma10,
            ma20,
            v_ma5,
            v_ma10,
            v_ma20,
            turnover,
            p_change
        ]

        features += on_feature

    return features


def get_mutil_all_features(stocks_list, feature_len, start_index, one_len=20):
    stock_days = []
    for stock in stocks_list:
        file_name = 'data/%s-%s-%s.csv' % (stock[2], stock[0], stock[1])
        if os.path.exists(file_name):
            stock_data = stock_load(file_name)
            temp_list = stock_mutil_history_feature(stock_data, feature_len, start_index)

            for feature, label in temp_list[:one_len]:
                if len(feature) == feature_len:
                    stock_days.append((stock[0], (feature, label)))

    print(len(stock_days))
    random.shuffle(stock_days)
    random.shuffle(stock_days)

    # print "stock_days:", stock_days[:1]
    return np.array([stock_feature(feature) + [label, code] for (code, (feature, label)) in stock_days])


def get_all_features(stocks_list, feature_len, start_index):
    stock_days = []
    for stock in stocks_list:
        file_name = 'data/%s-%s-%s.csv' % (stock[2], stock[0], stock[1])
        if os.path.exists(file_name):
            stock_data = stock_load(file_name)
            feature, label = stock_history_feature(stock_data, feature_len, start_index)
            if len(feature) == feature_len:
                stock_days.append((stock[0], (feature, label)))

    print(len(stock_days))
    random.shuffle(stock_days)

    # print "stock_days:", stock_days[:1]
    return np.array([stock_feature(feature) + [label, code] for (code, (feature, label)) in stock_days])


def train_model(test_start_index=0, cur_date=datetime.datetime.now().strftime('%Y%m%d')):
    """
    训练模型
    :param test_start_index:
    :param test_start_index:
    :param cur_date:
    :param industry_name: 行业名称
    """
    stock_industry = sd.stock_industry_load('data/stock_industry.csv')
    stocks_list = sd.industry_stock_all_list(stock_industry)
    stock_info_map = get_stock_info(stock_industry)
    # stock_open_set = stock_load_active()
    stock_open_set = stock_load_active_from_data(stocks_list)

    feature_len = 1
    start_index = test_start_index + 1
    test_start_index = test_start_index
    one_len = 100

    stock_feature_list = get_mutil_all_features(stocks_list, feature_len, start_index, one_len)

    temp_open_stock_list = [item for item in stocks_list if item[0] in stock_open_set]

    test_stock_feature_list = get_all_features(temp_open_stock_list, feature_len, test_start_index)


    print "open len:", len(stock_open_set)
    # print "xxx:",test_stock_feature_list[:-1]
    X = stock_feature_list[:, :-2]
    Y = stock_feature_list[:, -2]

    temp_test_X = test_stock_feature_list[:, :-2]
    temp_test_Y = test_stock_feature_list[:, -2]
    temp_test_index = test_stock_feature_list[:, -1]

    for item in test_stock_feature_list[:3]:
        print "test value:", item

    seed = 7

    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    print "all feature len:", len(X)
    print "train feature len:", len(X_train)
    print "test feature len:", len(X_test)

    params_fixed = {
        "objective": "binary:logistic",
        "silent": 1,
        "seed": seed
    }

    space = {
        "max_depth": (1, 5),
        "learning_rate": (10 ** -4, 10 ** -1),
        "n_estimators": (10, 200),
        "min_child_weight": (1, 20),
        "subsample": (0, 1),
        "colsample_bytree": (0.3, 1)
    }
    #
    # reg = XGBClassifier(**params_fixed)
    #
    # def objective(parms):
    #     reg.set_params(**{k: p for (k, p) in zip(space.keys(), parms)})
    #     return 1 - np.mean(cross_val_score(reg, X_train, y_train, cv=5, n_jobs=1, scoring="accuracy"))
    #
    # res_gp = gp_minimize(objective, space.values(), n_calls=50, random_state=seed)
    # best_hyper_params = {k: v for k, v in zip(space.keys(), res_gp.x)}
    #
    # print "best accuracy score = ", res_gp.fun
    # print "best paramers = ", best_hyper_params
    #
    # params = best_hyper_params.copy()
    # params.update(params_fixed)
    #
    # model = XGBClassifier(**params)
    #
    #
    # model.fit(X_train, y_train)
    #
    # y_test_preds = model.predict(X_test)
    #
    # pd.crosstab(
    #     pd.Series(y_test, name="Actual"),
    #     pd.Series(y_test_preds, name="Predicted"),
    #     margins=True
    # )
    #
    # print "Accuracy: {0:.3f}".format(accuracy_score(y_test, y_test_preds))


    model = XGBClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # predictions = [float(value) for value in y_pred]
    predictions = [value for value in y_pred]
    # for value in predictions:
    #     print "value:", value

    accuracy = accuracy_score(y_test, predictions)
    print("test Accuracy: %.2f%%" % (accuracy * 100.0))
    print("train Accuracy: %.2f%%" % (accuracy_score(y_train, model.predict(X_train)) * 100.0))

    temp_pred = model.predict_proba(X_test, True)
    for item in temp_pred[:3]:
        print "temp item:", item

    print "-------------------------"

    temp_y_pred = model.predict(X_test, True)
    for item in temp_y_pred[:3]:
        print "temp item:", item

    temp_pred_list = model.predict(temp_test_X)
    accuracy = accuracy_score(temp_test_Y, temp_pred_list)
    print("temp_test Accuracy: %.2f%%" % (accuracy * 100.0))

    temp_pred_list_pro = model.predict_proba(temp_test_X, True)

    temp_label_pred = [(0.0, label0) if label0 >= label1 else (1.0, label1) for label0, label1 in temp_pred_list_pro]

    temp_res_pred = [
        (temp_test_index[i], stock_info_map[temp_test_index[i]][1], temp_label_pred[i][0], temp_label_pred[i][1]) for i
        in range(0, len(temp_test_index))]

    pd.DataFrame(temp_res_pred, columns=["code", "name", "label", "score"]).to_csv(
        "./data/res_{0}.csv".format(cur_date))

    temp_incr = temp_res_pred
    temp_incr = filter(lambda x: x[2] >= 1.0, temp_incr)

    temp_incr_res = sorted(temp_incr, key=lambda a: a[3], reverse=True)

    for item in temp_incr_res[:20]:
        print "res item:", ",".join([str(v) for v in list(item)])


if __name__ == '__main__':
    test_start_index = 0
    if 2 == len(sys.argv):
        test_start_index = int(sys.argv[1].strip())
    train_model(test_start_index)
