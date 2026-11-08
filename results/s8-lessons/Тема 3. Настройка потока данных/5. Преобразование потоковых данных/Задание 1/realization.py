from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, TimestampType

# необходимая библиотека с идентификатором в maven
# вы можете использовать ее с помощью метода .config и опции "spark.jars.packages"
kafka_lib_id = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0"

# настройки security для кафки
# вы можете использовать из с помощью метода .options(**kafka_security_options)
kafka_security_options = {
    'kafka.security.protocol': 'SASL_SSL',
    'kafka.sasl.mechanism': 'SCRAM-SHA-512',
    'kafka.sasl.jaas.config': 'org.apache.kafka.common.security.scram.ScramLoginModule required username=\"de-student\" password=\"ltcneltyn\";',
}

# учебное имя входного топика для автоматической проверки
TOPIC_NAME = 'test-3-5-1'

value_schema = StructType([
    StructField("client_id", StringType()),
    StructField("timestamp", DoubleType()),
    StructField("lat", DoubleType()),
    StructField("lon", DoubleType()),
])


def spark_init() -> SparkSession:
    return (
        SparkSession.builder
        .master("local")
        .appName("test connect to kafka stream")
        .config("spark.jars.packages", kafka_lib_id)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def load_df(spark: SparkSession) -> DataFrame:
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", "rc1b-2erh7b35n4j4v869.mdb.yandexcloud.net:9091")
        .options(**kafka_security_options)
        .option("subscribe", TOPIC_NAME)
        .load()
    )


def transform(df: DataFrame) -> DataFrame:
    return (
        df
        .withColumn("value", f.col("value").cast(StringType()))
        .withColumn("value_json", f.from_json(f.col("value"), value_schema))
        .select(
            f.col("value_json.client_id").alias("client_id"),
            f.to_timestamp(
                f.from_unixtime(f.col("value_json.timestamp"))
            ).alias("timestamp"),
            f.col("value_json.lat").alias("lat"),
            f.col("value_json.lon").alias("lon"),
        )
    )


spark = spark_init()

source_df = load_df(spark)
output_df = transform(source_df)

query = (output_df
         .writeStream
         .outputMode("append")
         .format("console")
         .option("truncate", False)
         .trigger(once=True)
         .start())
try:
    query.awaitTermination()
finally:
    query.stop()
