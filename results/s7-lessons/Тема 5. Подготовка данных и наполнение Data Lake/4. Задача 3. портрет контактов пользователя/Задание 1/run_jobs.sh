#!/bin/sh
/usr/lib/spark/bin/spark-submit --master yarn --deploy-mode cluster connection_interests.py 2022-05-25 7 /user/prod/data/events /user/prod/data/analytics/user_interests_d7 /user/master/data/snapshots/tags_verified/actual /user/prod/data/analytics/connection_interests_d7
