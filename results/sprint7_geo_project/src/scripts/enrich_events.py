import sys

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from geo_common import attach_nearest_city, event_ts, event_user_id, load_cities


def main():
    date = sys.argv[1]
    events_path = sys.argv[2]
    geo_csv_path = sys.argv[3]
    output_path = sys.argv[4]

    spark = (
        SparkSession.builder.appName(f"GeoEnrichEvents-{date}")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )
    cities = load_cities(spark, geo_csv_path)
    events = spark.read.parquet(events_path)
    if date != "all":
        events = events.where(F.col("date") == date)

    events = events.withColumn("user_id", event_user_id()).withColumn("event_ts", event_ts())
    events = events.withColumn("row_id", F.monotonically_increasing_id())
    enriched = attach_nearest_city(events, cities)
    result = enriched.select(
        "row_id",
        "event",
        "event_type",
        "user_id",
        "event_ts",
        "lat",
        "lon",
        "zone_id",
        "city",
        "timezone",
        F.to_date("event_ts").alias("event_date"),
    )
    writer = result.coalesce(32).write.mode("overwrite")
    if date != "all":
        writer.partitionBy("event_type").parquet(f"{output_path}/date={date}")
    else:
        writer.partitionBy("event_type").parquet(output_path)
    spark.stop()


if __name__ == "__main__":
    main()
