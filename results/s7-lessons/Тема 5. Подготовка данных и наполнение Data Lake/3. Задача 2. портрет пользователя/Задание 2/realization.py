import datetime

import pyspark.sql.functions as F
from pyspark.sql.window import Window


def input_event_paths(date, depth):
    dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    return [
        f"/user/s27560962/data/events/date={(dt - datetime.timedelta(days=offset)).strftime('%Y-%m-%d')}/event_type=message"
        for offset in range(int(depth))
    ]


def reaction_tag_tops(date, depth, spark):
    message_paths = input_event_paths(date, depth)
    reaction_paths = [
        path.replace("/event_type=message", "/event_type=reaction")
        for path in message_paths
    ]
    posts = spark.read.parquet(*message_paths)
    reactions = spark.read.parquet(*reaction_paths)

    reaction_rows = reactions.select(
        F.col("event.reaction_from").alias("user_id"),
        F.col("event.message_id").alias("message_id"),
        F.col("event.reaction_type").alias("reaction_type"),
    ).where(F.col("user_id").isNotNull() & F.col("message_id").isNotNull())

    post_tags = (
        posts.select(
            F.col("event.message_id").alias("message_id"),
            F.explode("event.tags").alias("tag"),
        )
        .where(F.col("message_id").isNotNull())
        .distinct()
    )

    reacted = reaction_rows.join(post_tags, "message_id")
    w = Window.partitionBy("user_id").orderBy(F.col("tag_count").desc(), F.col("tag").desc())

    likes = (
        reacted.where(F.col("reaction_type") == "like")
        .groupBy("user_id", "tag")
        .agg(F.countDistinct("message_id").alias("tag_count"))
        .withColumn("rn", F.row_number().over(w))
        .where(F.col("rn") <= 3)
        .groupBy("user_id")
        .pivot("rn", [1, 2, 3])
        .agg(F.first("tag"))
        .select(
            "user_id",
            F.col("1").alias("like_tag_top_1"),
            F.col("2").alias("like_tag_top_2"),
            F.col("3").alias("like_tag_top_3"),
        )
    )
    dislikes = (
        reacted.where(F.col("reaction_type") == "dislike")
        .groupBy("user_id", "tag")
        .agg(F.countDistinct("message_id").alias("tag_count"))
        .withColumn("rn", F.row_number().over(w))
        .where(F.col("rn") <= 3)
        .groupBy("user_id")
        .pivot("rn", [1, 2, 3])
        .agg(F.first("tag"))
        .select(
            "user_id",
            F.col("1").alias("dislike_tag_top_1"),
            F.col("2").alias("dislike_tag_top_2"),
            F.col("3").alias("dislike_tag_top_3"),
        )
    )
    return likes.join(dislikes, "user_id", "full")
