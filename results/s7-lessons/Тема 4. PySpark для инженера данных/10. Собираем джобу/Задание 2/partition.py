import sys
 
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
 
def main():
        date = sys.argv[1]
        base_input_path = sys.argv[2]
        base_output_path = sys.argv[3]

        conf = SparkConf().setAppName(f"EventsPartitioningJob-{date}")
        sc = SparkContext(conf=conf)
        sql = SQLContext(sc)

        events = sql.read.json(f"{base_input_path}/date={date}")

        events\
        .write\
        .partitionBy('event_type')\
        .format('parquet')\
        .save(f"{base_output_path}/date={date}")


if __name__ == "__main__":
        main()
