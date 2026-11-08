import sys
import datetime

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
from pyspark.sql.window import Window


def input_event_paths(date, depth, base_path):
    days_count = depth
    dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    return [
        f"{base_path}/date={(dt - datetime.timedelta(days=offset)).strftime('%Y-%m-%d')}/event_type=message"
        for offset in range(int(days_count))
    ]


def tag_columns(reaction):
    return [f"{reaction}_tag_top_1", f"{reaction}_tag_top_2", f"{reaction}_tag_top_3"]


def _top3(df, prefix):
    ranked = (
        df.where(F.col("tag").isNotNull())
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
            F.col("1").alias(f"{prefix}_1"),
            F.col("2").alias(f"{prefix}_2"),
            F.col("3").alias(f"{prefix}_3"),
        )
    )


def get_contacts(direct_messages):
    dm = direct_messages.where(
        F.col("event.message_from").isNotNull() & F.col("event.message_to").isNotNull()
    )
    outgoing = dm.select(
        F.col("event.message_from").cast("string").alias("user_id"),
        F.col("event.message_to").cast("string").alias("contact_id"),
    )
    incoming = dm.select(
        F.col("event.message_to").cast("string").alias("user_id"),
        F.col("event.message_from").cast("string").alias("contact_id"),
    )
    return outgoing.union(incoming).where(F.col("user_id") != F.col("contact_id")).distinct()


def get_contact_interests(contacts, interests):
    contact_profile = interests.withColumn(
        "user_id", F.col("user_id").cast("string")
    ).withColumnRenamed("user_id", "contact_id")
    joined = contacts.join(contact_profile, "contact_id")
    likes = _top3(
        joined.select("user_id", F.explode(F.array(*tag_columns("like"))).alias("tag")),
        "direct_like_tag_top",
    )
    dislikes = _top3(
        joined.select("user_id", F.explode(F.array(*tag_columns("dislike"))).alias("tag")),
        "direct_dislike_tag_top",
    )
    return likes.join(dislikes, "user_id", "full")


def get_subs_interests(posts, subscriptions, verified_tags):
    channel_tags = (
        posts
        .where(F.col("event.message_channel_to").isNotNull())
        .select(
            F.col("event.message_channel_to").alias("channel_id"),
            F.explode("event.tags").alias("tag"),
        )
        .where(F.col("tag").isNotNull())
    )
    subscribers = (
        subscriptions
        .where(
            F.col("event.subscription_channel").isNotNull()
            & F.col("event.user").isNotNull()
        )
        .select(
            F.col("event.user").cast("string").alias("user_id"),
            F.col("event.subscription_channel").alias("channel_id"),
        )
        .distinct()
    )
    user_tags = (
        subscribers
        .join(channel_tags, "channel_id")
        .join(verified_tags, "tag", "inner")
        .select("user_id", "tag")
    )
    return _top3(user_tags, "sub_verified_tag_top")


def join_result(contact_interests, subs_interests):
    return contact_interests.join(subs_interests, "user_id", "full")


def main():
    date = sys.argv[1]
    days_count = sys.argv[2]
    events_base_path = sys.argv[3]
    interests_base_path = sys.argv[4]
    verified_tags_path = sys.argv[5]
    output_base_path = sys.argv[6]

    conf = SparkConf().setAppName(f"ConnectionInterestsJob-{date}-d{days_count}")
    sc = SparkContext(conf=conf)
    spark = SQLContext(sc)

    messages = spark.read.parquet(*input_event_paths(date, days_count, events_base_path))
    direct_messages = messages.where(F.col("event.message_to").isNotNull())
    posts = messages.where(F.col("event.message_channel_to").isNotNull())
    interests = spark.read.parquet(f"{interests_base_path}/date={date}")
    subscriptions = spark.read.parquet(f"{events_base_path}/*/event_type=subscription")
    verified_tags = spark.read.parquet(verified_tags_path)
    contacts = get_contacts(direct_messages)
    contact_interests = get_contact_interests(contacts, interests)
    subs_interests = get_subs_interests(posts, subscriptions, verified_tags)
    result = join_result(contact_interests, subs_interests)
    result.write.parquet(f"{output_base_path}/date={date}")


if __name__ == "__main__":
    main()
