import pyspark.sql.functions as F
from pyspark.sql import DataFrame


CITY_TIMEZONES = {
    "Sydney": "Australia/Sydney",
    "Melbourne": "Australia/Melbourne",
    "Brisbane": "Australia/Brisbane",
    "Perth": "Australia/Perth",
    "Adelaide": "Australia/Adelaide",
    "Gold Coast": "Australia/Brisbane",
    "Cranbourne": "Australia/Melbourne",
    "Canberra": "Australia/Sydney",
    "Newcastle": "Australia/Sydney",
    "Wollongong": "Australia/Sydney",
    "Geelong": "Australia/Melbourne",
    "Hobart": "Australia/Hobart",
    "Townsville": "Australia/Brisbane",
    "Ipswich": "Australia/Brisbane",
    "Cairns": "Australia/Brisbane",
    "Toowoomba": "Australia/Brisbane",
    "Darwin": "Australia/Darwin",
    "Ballarat": "Australia/Melbourne",
    "Bendigo": "Australia/Melbourne",
    "Launceston": "Australia/Hobart",
    "Mackay": "Australia/Brisbane",
    "Rockhampton": "Australia/Brisbane",
    "Maitland": "Australia/Sydney",
    "Bunbury": "Australia/Perth",
}


def haversine_km(lat1, lon1, lat2, lon2):
    return (
        F.lit(2 * 6371)
        * F.asin(
            F.sqrt(
                F.pow(F.sin((F.radians(lat2) - F.radians(lat1)) / 2), 2)
                + F.cos(F.radians(lat1))
                * F.cos(F.radians(lat2))
                * F.pow(F.sin((F.radians(lon2) - F.radians(lon1)) / 2), 2)
            )
        )
    )


def load_cities(spark, geo_csv_path: str) -> DataFrame:
    cities = (
        spark.read.option("header", True)
        .option("sep", ";")
        .csv(geo_csv_path)
        .withColumn("zone_id", F.col("id").cast("int"))
        .withColumn("city", F.trim(F.col("city")))
        .withColumn("lat_city", F.regexp_replace("lat", ",", ".").cast("double"))
        .withColumn("lon_city", F.regexp_replace("lng", ",", ".").cast("double"))
    )
    tz_rows = [(city, tz) for city, tz in CITY_TIMEZONES.items()]
    tz_df = spark.createDataFrame(tz_rows, ["city", "timezone"])
    return cities.join(tz_df, "city", "left").select(
        "zone_id", "city", "lat_city", "lon_city", "timezone"
    )


def event_user_id():
    return F.coalesce(
        F.col("event.message_from").cast("long"),
        F.col("event.reaction_from").cast("long"),
        F.col("event.user").cast("long"),
    ).alias("user_id")


def event_ts():
    return F.coalesce(
        F.to_timestamp("event.message_ts"),
        F.to_timestamp("event.datetime"),
    ).alias("event_ts")


def attach_nearest_city(events: DataFrame, cities: DataFrame) -> DataFrame:
    from pyspark.sql.window import Window

    with_coords = events.where(F.col("lat").isNotNull() & F.col("lon").isNotNull())
    paired = with_coords.crossJoin(F.broadcast(cities))
    ranked = paired.withColumn(
        "distance_km",
        haversine_km(F.col("lat"), F.col("lon"), F.col("lat_city"), F.col("lon_city")),
    )
    key = "row_id" if "row_id" in events.columns else "event"
    w = Window.partitionBy(key).orderBy("distance_km", "zone_id")
    return (
        ranked.withColumn("rn", F.row_number().over(w))
        .where(F.col("rn") == 1)
        .drop("rn", "lat_city", "lon_city", "distance_km")
    )
