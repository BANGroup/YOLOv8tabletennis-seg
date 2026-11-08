import datetime

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window


def input_paths(date, depth):
    dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    return [
        f"/user/s27560962/data/events/date={(dt - datetime.timedelta(days=offset)).strftime('%Y-%m-%d')}/event_type=message"
        for offset in range(depth)
    ]


spark = SparkSession.builder \
    .master("yarn") \
    .appName("SuggestedTagsD7") \
    .getOrCreate()

messages = spark.read.parquet(*input_paths("2022-05-31", 7))
all_tags = (
    messages
    .where("event.message_channel_to is not null")
    .selectExpr("event.message_from as user", "explode(event.tags) as tag")
)
verified_tags = spark.read.parquet("/user/master/data/snapshots/tags_verified/actual")

candidates = (
    all_tags
    .groupBy("tag")
    .agg(F.countDistinct("user").alias("suggested_count"))
    .where("suggested_count >= 100")
    .join(verified_tags, "tag", "left_anti")
)

candidates.write.mode("overwrite").parquet("/user/s27560962/data/analytics/candidates_d7_pyspark")
