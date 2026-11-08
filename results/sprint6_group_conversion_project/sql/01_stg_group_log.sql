DROP TABLE IF EXISTS VT2609246827BE__STAGING.group_log CASCADE;

CREATE TABLE VT2609246827BE__STAGING.group_log
(
    group_id int NOT NULL,
    user_id int NOT NULL,
    user_id_from int,
    event varchar(20),
    datetime timestamp,
    CONSTRAINT C_PRIMARY PRIMARY KEY (group_id, user_id, datetime, event) DISABLED
)
ORDER BY group_id, user_id
SEGMENTED BY hash(group_id, user_id) ALL NODES;
