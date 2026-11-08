from pyspark.sql import SparkSession
spark = SparkSession \
        .builder \
        .config("spark.driver.memory", "1g") \
        .config("spark.driver.cores", "2") \
        .appName("My first session") \
        .getOrCreate()
