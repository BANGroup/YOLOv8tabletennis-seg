#!/bin/bash
set -euo pipefail

# Тестовая отправка акции в входной топик. restaurant_id совпадает с subscribers_restaurants.
# start/end — текущее UTC ± 1 час, чтобы фильтр «акция действует» пропустил сообщение.

TOPIC_IN="${TOPIC_IN:-s27560962_in}"
BOOTSTRAP="${BOOTSTRAP:-rc1b-2erh7b35n4j4v869.mdb.yandexcloud.net:9091}"
KAFKA_USER="${KAFKA_USER:-de-student}"
KAFKA_PASSWORD="${KAFKA_PASSWORD:-ltcneltyn}"

NOW="$(date -u +%s)"
START="$((NOW - 3600))"
END="$((NOW + 3600))"

KEY="${1:-1}"
PAYLOAD=$(cat <<EOF
{"restaurant_id": "123e4567-e89b-12d3-a456-426614174000","adv_campaign_id": "123e4567-e89b-12d3-a456-426614174003","adv_campaign_content": "first campaign","adv_campaign_owner": "Ivanov Ivan Ivanovich","adv_campaign_owner_contact": "iiivanov@restaurant.ru","adv_campaign_datetime_start": ${START},"adv_campaign_datetime_end": ${END},"datetime_created": ${NOW}}
EOF
)

printf '%s:%s\n' "${KEY}" "${PAYLOAD}" | kcat -b "${BOOTSTRAP}" \
    -X security.protocol=SASL_SSL \
    -X sasl.mechanisms=SCRAM-SHA-512 \
    -X sasl.username="${KAFKA_USER}" \
    -X sasl.password="${KAFKA_PASSWORD}" \
    -t "${TOPIC_IN}" \
    -K: \
    -P

echo "sent ${KEY} to ${TOPIC_IN} start=${START} end=${END}"
