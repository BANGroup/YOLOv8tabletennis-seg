from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, TimestampType, IntegerType


def spark_init(test_name) -> SparkSession:
    spark_jars_packages = ",".join(
        [
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0",
            "org.postgresql:postgresql:42.4.0",
        ]
    )
    return (
        SparkSession.builder.appName(test_name)
        .config("spark.jars.packages", spark_jars_packages)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

postgresql_settings = {
    'user': 'master',
    'password': 'de-master-password'
}


def read_marketing(spark: SparkSession) -> DataFrame:
    return (
        spark.read
        .format("jdbc")
        .option("url", "jdbc:postgresql://rc1a-fswjkpli01zafgjm.mdb.yandexcloud.net:6432/de")
        .option("driver", "org.postgresql.Driver")
        .option("dbtable", "public.marketing_companies")
        .options(**postgresql_settings)
        .load()
    )



kafka_security_options = {
    'kafka.security.protocol': 'SASL_SSL',
    'kafka.sasl.mechanism': 'SCRAM-SHA-512',
    'kafka.sasl.jaas.config': 'org.apache.kafka.common.security.scram.ScramLoginModule required username=\"de-student\" password=\"ltcneltyn\";',
}


def read_client_stream(spark: SparkSession) -> DataFrame:
    value_schema = StructType([
        StructField("client_id", StringType()),
        StructField("timestamp", DoubleType()),
        StructField("lat", DoubleType()),
        StructField("lon", DoubleType()),
    ])
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", "rc1b-2erh7b35n4j4v869.mdb.yandexcloud.net:9091")
        .options(**kafka_security_options)
        .option("subscribe", "test-3-5-1")
        .load()
        .withColumn("value", f.col("value").cast(StringType()))
        .withColumn("event", f.from_json(f.col("value"), value_schema))
        .select(
            f.col("event.client_id").alias("client_id"),
            f.to_timestamp(f.from_unixtime(f.col("event.timestamp"))).alias("timestamp"),
            f.col("event.lat").alias("lat"),
            f.col("event.lon").alias("lon"),
            f.col("offset"),
        )
        .withWatermark("timestamp", "10 minutes")
        .dropDuplicates(["client_id", "timestamp"])
    )


def join(user_df, marketing_df) -> DataFrame:
    return (
        user_df.crossJoin(marketing_df)
        .select(
            user_df.client_id.alias("client_id"),
            marketing_df.id.alias("adv_campaign_id"),
            marketing_df.name.alias("adv_campaign_name"),
            marketing_df.description.alias("adv_campaign_description"),
            marketing_df.start_time.alias("adv_campaign_start_time"),
            marketing_df.end_time.alias("adv_campaign_end_time"),
            marketing_df.point_lat.alias("adv_campaign_point_lat"),
            marketing_df.point_lon.alias("adv_campaign_point_lon"),
            f.current_timestamp().alias("created_at"),
            user_df.offset.alias("offset"),
        )
    )


if __name__ == "__main__":
    spark = spark_init('join stream')
    client_stream = read_client_stream(spark)
    marketing_df = read_marketing(spark)
    result = join(client_stream, marketing_df)

    query = (result
             .writeStream
             .outputMode("append")
             .format("console")
             .option("truncate", False)
             .start())
    query.awaitTermination()
