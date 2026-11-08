import sys

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window


def main():
    events_ods_path = sys.argv[1]
    output_path = sys.argv[2]

    spark = SparkSession.builder.appName("GeoUserDatamart").getOrCreate()
    events = (
        spark.read.parquet(events_ods_path)
        .where(F.col("user_id").isNotNull() & F.col("city").isNotNull() & F.col("event_ts").isNotNull())
    )

    by_time = Window.partitionBy("user_id").orderBy(F.col("event_ts").desc())
    latest = (
        events.withColumn("rn", F.row_number().over(by_time))
        .where(F.col("rn") == 1)
        .select(
            "user_id",
            F.col("city").alias("act_city"),
            F.from_utc_timestamp("event_ts", F.col("timezone")).alias("local_time"),
        )
    )

    daily = (
        events.withColumn(
            "day_rn",
            F.row_number().over(Window.partitionBy("user_id", "event_date").orderBy(F.col("event_ts").desc())),
        )
        .where(F.col("day_rn") == 1)
        .select("user_id", "event_date", "city")
    )
    day_w = Window.partitionBy("user_id").orderBy("event_date")
    stays = (
        daily.withColumn("prev_city", F.lag("city").over(day_w))
        .withColumn("is_new_stay", F.when(F.col("city") != F.col("prev_city"), F.lit(1)).otherwise(F.lit(0)))
        .withColumn("stay_id", F.sum("is_new_stay").over(day_w.rowsBetween(Window.unboundedPreceding, 0)))
        .groupBy("user_id", "stay_id", "city")
        .agg(
            F.min("event_date").alias("stay_from"),
            F.max("event_date").alias("stay_to"),
        )
        .withColumn("stay_days", F.datediff("stay_to", "stay_from") + 1)
    )
    home_city = (
        stays.where(F.col("stay_days") >= 27)
        .withColumn("rn", F.row_number().over(Window.partitionBy("user_id").orderBy(F.col("stay_to").desc())))
        .where(F.col("rn") == 1)
        .select("user_id", F.col("city").alias("home_city"))
    )

    travel_w = Window.partitionBy("user_id").orderBy("event_ts")
    travels = (
        events.withColumn("prev_city", F.lag("city").over(travel_w))
        .where(F.col("prev_city").isNull() | (F.col("city") != F.col("prev_city")))
        .groupBy("user_id")
        .agg(
            F.count("*").alias("travel_count"),
            F.collect_list("city").alias("travel_array"),
        )
    )

    result = (
        latest.join(home_city, "user_id", "left")
        .join(travels, "user_id", "left")
        .select(
            "user_id",
            "act_city",
            "home_city",
            "travel_count",
            "travel_array",
            "local_time",
        )
    )
    result.write.mode("overwrite").parquet(output_path)
    spark.stop()


if __name__ == "__main__":
    main()
