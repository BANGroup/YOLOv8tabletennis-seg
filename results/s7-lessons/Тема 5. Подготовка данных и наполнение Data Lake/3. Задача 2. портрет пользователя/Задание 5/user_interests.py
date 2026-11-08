import sys
import datetime

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
from pyspark.sql.window import Window


def input_event_paths(base_path, date, depth):
    dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    return [
        f"{base_path}/date={(dt - datetime.timedelta(days=offset)).strftime('%Y-%m-%d')}/event_type=message"
        for offset in range(int(depth))
    ]


def tag_tops(posts):
    user_tags = (
        posts
        .where(F.col("event.message_from").isNotNull())
        .select(
            F.col("event.message_from").alias("user_id"),
            F.col("event.message_id").alias("message_id"),
            F.explode("event.tags").alias("tag"),
        )
    )
    ranked = (
        user_tags
        .groupBy("user_id", "tag")
        .agg(F.count("*").alias("tag_count"))
        .withColumn(
            "rn",
            F.row_number().over(
                Window.partitionBy("user_id").orderBy(
                    F.col("tag_count").desc(),
                    F.col("tag").desc(),
                )
            ),
        )
        .where(F.col("rn") <= 3)
    )
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


def reaction_tag_tops(posts_all, reactions):
    reaction_rows = reactions.select(
        F.col("event.reaction_from").alias("user_id"),
        F.col("event.message_id").alias("message_id"),
        F.col("event.reaction_type").alias("reaction_type"),
    ).where(F.col("user_id").isNotNull() & F.col("message_id").isNotNull())

    post_tags = (
        posts_all.select(
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


def calculate_user_interests(posts_part, posts_all, reactions):
    return tag_tops(posts_part).join(
        reaction_tag_tops(posts_all, reactions),
        "user_id",
        "full",
    )


def main():
    date = sys.argv[1]
    days_count = sys.argv[2]
    events_base_path = sys.argv[3]
    output_base_path = sys.argv[4]

    conf = SparkConf().setAppName(f"UserInterestsJob-{date}-d{days_count}")
    sc = SparkContext(conf=conf)
    sql = SQLContext(sc)

    message_paths = input_event_paths(events_base_path, date, days_count)
    reaction_paths = [
        path.replace("/event_type=message", "/event_type=reaction")
        for path in message_paths
    ]
    posts_part = sql.read.parquet(*message_paths)
    reactions = sql.read.parquet(*reaction_paths)
    posts_all = sql.read.parquet(f"{events_base_path}/*/event_type=message")

    result = calculate_user_interests(posts_part, posts_all, reactions)
    result.write.parquet(f"{output_base_path}/date={date}")


if __name__ == "__main__":
    main()
