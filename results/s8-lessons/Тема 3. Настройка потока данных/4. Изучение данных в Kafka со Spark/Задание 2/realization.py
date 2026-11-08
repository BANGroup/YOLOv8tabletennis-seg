from pyspark.sql import SparkSession
from pyspark.sql import functions as f, DataFrame
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType

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

value_schema = StructType([
    StructField("subscription_id", IntegerType()),
    StructField("name", StringType()),
    StructField("description", StringType()),
    StructField("price", DoubleType()),
    StructField("currency", StringType()),
])


def spark_init() -> SparkSession:
    return (
        SparkSession.builder
        .master("local")
        .appName("test connect to kafka")
        .config("spark.jars.packages", kafka_lib_id)
        .getOrCreate()
    )


def load_df(spark: SparkSession) -> DataFrame:
    return (
        spark.read
        .format("kafka")
        .option("kafka.bootstrap.servers", "rc1b-2erh7b35n4j4v869.mdb.yandexcloud.net:9091")
        .options(**kafka_security_options)
        .option("subscribe", "persist_topic")
        .load()
    )


def transform(df: DataFrame) -> DataFrame:
    return (
        df
        .withColumn("key", f.col("key").cast(StringType()))
        .withColumn("value", f.col("value").cast(StringType()))
        .withColumn("value_json", f.from_json(f.col("value"), value_schema))
        .select(
            f.col("value_json.subscription_id").alias("subscription_id"),
            f.col("value_json.name").alias("name"),
            f.col("value_json.description").alias("description"),
            f.col("value_json.price").alias("price"),
            f.col("value_json.currency").alias("currency"),
            "key",
            "value",
            "topic",
            "partition",
            "offset",
            "timestamp",
            "timestampType",
        )
    )


spark = spark_init()

source_df = load_df(spark)
df = transform(source_df)


df.printSchema()
df.show(truncate=False)
