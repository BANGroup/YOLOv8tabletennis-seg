import datetime

import pyspark.sql.functions as F
from pyspark.sql.window import Window


def input_event_paths(date, depth):
    dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    return [
        f"/user/s27560962/data/events/date={(dt - datetime.timedelta(days=offset)).strftime('%Y-%m-%d')}/event_type=message"
        for offset in range(int(depth))
    ]


def tag_tops(date, depth, spark):
    messages = spark.read.parquet(*input_event_paths(date, depth))
    user_tags = (
        messages
        .where(F.col("event.message_from").isNotNull())
        .select(
            F.col("event.message_from").alias("user_id"),
            F.col("event.message_id").alias("message_id"),
            F.explode("event.tags").alias("tag"),
        )
    )
    tag_counts = (
        user_tags
        .groupBy("user_id", "tag")
        .agg(F.count("*").alias("tag_count"))
    )
    ranked = tag_counts.withColumn(
        "rn",
        F.row_number().over(
            Window.partitionBy("user_id").orderBy(
                F.col("tag_count").desc(),
                F.col("tag").desc(),
            )
        ),
    ).where(F.col("rn") <= 3)
    return (
        ranked
        .groupBy("user_id")
        .pivot("rn", [1, 2, 3])
        .agg(F.first("tag"))
        .select(
            "user_id",
            F.col("1").alias("tag_top_1"),
            F.col("2").alias("tag_top_2"),
            F.col("3").alias("tag_top_3"),
        )
    )
