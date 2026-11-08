import sys

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from geo_common import haversine_km


def main():
    events_ods_path = sys.argv[1]
    output_path = sys.argv[2]

    spark = (
        SparkSession.builder.appName("GeoFriendsDatamart")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )
    events = spark.read.parquet(events_ods_path)

    last_location = (
        events.where(
            (F.col("event_type") == "message")
            & F.col("user_id").isNotNull()
            & F.col("lat").isNotNull()
            & F.col("lon").isNotNull()
        )
        .withColumn("rn", F.row_number().over(Window.partitionBy("user_id").orderBy(F.col("event_ts").desc())))
        .where(F.col("rn") == 1)
        .select("user_id", "lat", "lon", "zone_id", "timezone", "event_ts")
    )

    last_location = last_location.withColumn(
        "lat_cell", F.floor(F.col("lat") * F.lit(111.0))
    ).withColumn("lon_cell", F.floor(F.col("lon") * F.lit(111.0)))
    left = last_location.select(
        F.col("user_id").alias("user_left"),
        F.col("lat").alias("lat_left"),
        F.col("lon").alias("lon_left"),
        F.col("lat_cell").alias("lat_cell_l"),
        F.col("lon_cell").alias("lon_cell_l"),
        F.col("zone_id"),
        F.col("timezone"),
    )
    right = last_location.select(
        F.col("user_id").alias("user_right"),
        F.col("lat").alias("lat_right"),
        F.col("lon").alias("lon_right"),
        F.col("lat_cell").alias("lat_cell_r"),
        F.col("lon_cell").alias("lon_cell_r"),
    )
    left_cells = (
        left.withColumn("dlat", F.explode(F.array(F.lit(-1), F.lit(0), F.lit(1))))
        .withColumn("dlon", F.explode(F.array(F.lit(-1), F.lit(0), F.lit(1))))
        .withColumn("lat_cell_r", F.col("lat_cell_l") + F.col("dlat"))
        .withColumn("lon_cell_r", F.col("lon_cell_l") + F.col("dlon"))
    )
    nearby = (
        left_cells.join(right, ["lat_cell_r", "lon_cell_r"]).where(F.col("user_left") < F.col("user_right"))
        .withColumn(
            "distance_km",
            haversine_km(F.col("lat_left"), F.col("lon_left"), F.col("lat_right"), F.col("lon_right")),
        )
        .where(F.col("distance_km") <= 1)
        .select("user_left", "user_right", "zone_id", "timezone")
    )

    messages = (
        events.where(
            (F.col("event_type") == "message")
            & F.col("event.message_from").isNotNull()
            & F.col("event.message_to").isNotNull()
        )
        .select(
            F.col("event.message_from").cast("long").alias("user_left"),
            F.col("event.message_to").cast("long").alias("user_right"),
        )
    )
    talked = messages.unionByName(
        messages.select(F.col("user_right").alias("user_left"), F.col("user_left").alias("user_right"))
    ).distinct()
    no_chat = nearby.join(talked, ["user_left", "user_right"], "left_anti")

    subs = (
        events.where(
            (F.col("event_type") == "subscription")
            & F.col("event.subscription_channel").isNotNull()
            & F.col("user_id").isNotNull()
        )
        .select("user_id", F.col("event.subscription_channel").alias("channel_id"))
        .distinct()
    )
    shared = (
        subs.alias("a")
        .join(subs.alias("b"), F.col("a.channel_id") == F.col("b.channel_id"))
        .where(F.col("a.user_id") < F.col("b.user_id"))
        .select(
            F.col("a.user_id").alias("user_left"),
            F.col("b.user_id").alias("user_right"),
        )
        .distinct()
    )

    result = (
        no_chat.join(shared, ["user_left", "user_right"])
        .withColumn("processed_dttm", F.current_timestamp())
        .withColumn("local_time", F.from_utc_timestamp(F.col("processed_dttm"), F.col("timezone")))
        .select("user_left", "user_right", "processed_dttm", "zone_id", "local_time")
    )
    result.write.mode("overwrite").parquet(output_path)
    spark.stop()


if __name__ == "__main__":
    main()
