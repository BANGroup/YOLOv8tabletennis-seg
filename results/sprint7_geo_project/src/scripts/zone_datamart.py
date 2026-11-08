import sys

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window


def main():
    events_ods_path = sys.argv[1]
    output_path = sys.argv[2]

    spark = SparkSession.builder.appName("GeoZoneDatamart").getOrCreate()
    events = (
        spark.read.parquet(events_ods_path)
        .where(F.col("zone_id").isNotNull() & F.col("event_ts").isNotNull())
        .withColumn("month", F.trunc("event_ts", "month"))
        .withColumn("week", F.date_trunc("week", "event_ts"))
    )

    week_counts = (
        events.groupBy("month", "week", "zone_id")
        .pivot("event_type", ["message", "reaction", "subscription"])
        .count()
        .fillna(0)
        .select(
            "month",
            "week",
            "zone_id",
            F.col("message").alias("week_message"),
            F.col("reaction").alias("week_reaction"),
            F.col("subscription").alias("week_subscription"),
        )
    )
    month_counts = (
        events.groupBy("month", "zone_id")
        .pivot("event_type", ["message", "reaction", "subscription"])
        .count()
        .fillna(0)
        .select(
            "month",
            "zone_id",
            F.col("message").alias("month_message"),
            F.col("reaction").alias("month_reaction"),
            F.col("subscription").alias("month_subscription"),
        )
    )

    first_events = (
        events.where(F.col("user_id").isNotNull())
        .withColumn("rn", F.row_number().over(Window.partitionBy("user_id").orderBy("event_ts")))
        .where(F.col("rn") == 1)
    )
    week_users = first_events.groupBy("month", "week", "zone_id").agg(F.countDistinct("user_id").alias("week_user"))
    month_users = first_events.groupBy("month", "zone_id").agg(F.countDistinct("user_id").alias("month_user"))

    result = (
        week_counts.join(week_users, ["month", "week", "zone_id"], "left")
        .join(month_counts, ["month", "zone_id"], "left")
        .join(month_users, ["month", "zone_id"], "left")
        .fillna(0)
        .select(
            "month",
            "week",
            "zone_id",
            "week_message",
            "week_reaction",
            "week_subscription",
            "week_user",
            "month_message",
            "month_reaction",
            "month_subscription",
            "month_user",
        )
    )
    result.write.mode("overwrite").parquet(output_path)
    spark.stop()


if __name__ == "__main__":
    main()
