import datetime

import pyspark.sql.functions as F
from pyspark.sql.window import Window


def compare_df(left, right):
    if set(left.columns) != set(right.columns):
        return False
    aligned = right.select(*left.columns)
    return left.exceptAll(aligned).count() == 0 and aligned.exceptAll(left).count() == 0
