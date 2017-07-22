# coding=utf-8
import datetime

IOSDATEFROMAT = "%Y-%m-%d %H:%M:%S"
YMDDATAFROMAT = "%Y-%m-%d"


def get_before_day(len):
    """
    获取前几天的日期
    :param len:
    :return:
    """
    now = datetime.datetime.now()
    delta = datetime.timedelta(days=-len)
    before_day = now + delta
    return before_day.strftime(YMDDATAFROMAT)
