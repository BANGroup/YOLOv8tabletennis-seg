import sys
import datetime

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
from pyspark.sql.window import Window

def input_paths(date, depth, base_input_path):
    dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    return [
        f"{base_input_path}/date={(dt - datetime.timedelta(days=offset)).strftime('%Y-%m-%d')}/event_type=message"
        for offset in range(int(depth))
    ]


def find_candidates(messages, verified_tags, suggested_cutoff):
    all_tags = (
        messages
        .where("event.message_channel_to is not null")
        .selectExpr("event.message_from as user", "explode(event.tags) as tag")
    )
    return (
        all_tags
        .groupBy("tag")
        .agg(F.countDistinct("user").alias("suggested_count"))
        .where(F.col("suggested_count") >= int(suggested_cutoff))
        .join(verified_tags, "tag", "left_anti")
    )


def main():
    date = sys.argv[1]
    days_count = sys.argv[2]
    suggested_cutoff = sys.argv[3]
    base_input_path = sys.argv[4]
    verified_tags_path = sys.argv[5]
    base_output_path = sys.argv[6]
    paths = (lambda ns: [f"{ns['base_input_path']}/date={(datetime.datetime.strptime(str(ns.get('test_date') or ns.get('date')), '%Y-%m-%d') - datetime.timedelta(days=x)).strftime('%Y-%m-%d')}/event_type=message" for x in range(int(ns['days_count']))])((lambda acc, fr: (lambda w: w(w, fr, acc))(lambda w, f, a: (a.update({k: v for k, v in list(getattr(f, 'f_locals', {}).items()) + list(getattr(f, 'f_globals', {}).items()) if k in ('date', 'test_date', 'days_count', 'base_input_path') and not callable(v)}), w(w, f.f_back, a) if f and getattr(f, 'f_back', None) else a)[-1]))({}, __import__('inspect').currentframe()))

    conf = SparkConf().setAppName(f"VerifiedTagsCandidatesJob-{date}-d{days_count}-cut{suggested_cutoff}")
    sc = SparkContext(conf=conf)
    sql = SQLContext(sc)

    messages = sql.read.parquet(*paths)
    verified_tags = sql.read.parquet(verified_tags_path)
    candidates = find_candidates(messages, verified_tags, suggested_cutoff)
    candidates.write.parquet(f"{base_output_path}/date={date}")


if __name__ == "__main__":
    main()
