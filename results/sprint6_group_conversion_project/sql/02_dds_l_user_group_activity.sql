DROP TABLE IF EXISTS VT2609246827BE__DWH.l_user_group_activity CASCADE;

CREATE TABLE VT2609246827BE__DWH.l_user_group_activity
(
    hk_l_user_group_activity int NOT NULL,
    hk_user_id int NOT NULL,
    hk_group_id int NOT NULL,
    load_dt timestamp,
    load_src varchar(20),
    CONSTRAINT C_PRIMARY PRIMARY KEY (hk_l_user_group_activity) DISABLED
)
PARTITION BY ((l_user_group_activity.load_dt)::date)
GROUP BY calendar_hierarchy_day((l_user_group_activity.load_dt)::date, 3, 2);

ALTER TABLE VT2609246827BE__DWH.l_user_group_activity
    ADD CONSTRAINT fk_l_user_group_activity_user
    FOREIGN KEY (hk_user_id) REFERENCES VT2609246827BE__DWH.h_users (hk_user_id);

ALTER TABLE VT2609246827BE__DWH.l_user_group_activity
    ADD CONSTRAINT fk_l_user_group_activity_group
    FOREIGN KEY (hk_group_id) REFERENCES VT2609246827BE__DWH.h_groups (hk_group_id);

CREATE PROJECTION VT2609246827BE__DWH.l_user_group_activity
(
    hk_l_user_group_activity,
    hk_user_id,
    hk_group_id,
    load_dt,
    load_src
)
AS
SELECT
    l_user_group_activity.hk_l_user_group_activity,
    l_user_group_activity.hk_user_id,
    l_user_group_activity.hk_group_id,
    l_user_group_activity.load_dt,
    l_user_group_activity.load_src
FROM VT2609246827BE__DWH.l_user_group_activity
ORDER BY l_user_group_activity.load_dt
SEGMENTED BY l_user_group_activity.hk_l_user_group_activity ALL NODES;
