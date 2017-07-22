# coding=utf-8

import sys
import tushare as ts

reload(sys)
sys.setdefaultencoding("utf-8")

ts.set_token("562b2fdd786905ab891533d09b42c4cb65f30ad7ec80fb6fc3f7733309ce810f")

def stock_industry_download():
    """
    下载股票行业分类数据
    """
    df = ts.get_industry_classified()
    df.to_csv('./data/stock_industry.csv')

def stock_open_data_download():
    # 获取沪深A股正常股票信息，listStatusCD上市状态，可选状态有L——上市，S——暂停，DE——已退市，UN——未上市
    eq = ts.Equity()
    df = eq.Equ(equTypeCD='A', listStatusCD='L', field='ticker,secShortName,totalShares,nonrestFloatShares')
    df['ticker'] = df['ticker'].map(lambda x: str(x).zfill(6))
    df.to_csv("./data/stock_open.csv")

def stock_industry_load(file_name):
    """
    加载股票行业分类信息
    :param file_name: 文件名
    :return: {'行业名':[('股票代码', '股票名', '行业名'), ...]}
    """
    stock_industry = {}
    file = open(file_name)

    is_first_line = True
    for line in file:
        if is_first_line:
            is_first_line = False
            continue
        prop_list = line.split(',')
        industry_name = prop_list[3].strip()
        if industry_name not in stock_industry:
            stock_industry[industry_name] = []
        stock_industry[industry_name].append((prop_list[1], prop_list[2], industry_name))

    file.close()
    return stock_industry


def industry_stock_all_list(industry_data):
    res = []
    for item in industry_data:
        res += industry_data[item]

    return res


def industry_stock_download(name=None):
    """
    下载某一行业的所有股票数据
    :param name: 行业名称
    """
    industry_data = stock_industry_load('./data/stock_industry.csv')

    temp_list = industry_data[name] if name else industry_stock_all_list(industry_data)

    # for stock in industry_data[name]:
    for stock in temp_list:
        print stock
        stock_code = stock[0]
        stock_name = stock[1]
        industry_name = stock[2]
        print(stock_code)
        print(stock_name)

        df = ts.get_hist_data(stock_code)
        if df is not None:
            df.to_csv('data/%s-%s-%s.csv' % (industry_name, stock_code, stock_name))


if __name__ == '__main__':
    stock_open_data_download()
    stock_industry_download()
    # industry_stock_download('电子信息')
    industry_stock_download()
