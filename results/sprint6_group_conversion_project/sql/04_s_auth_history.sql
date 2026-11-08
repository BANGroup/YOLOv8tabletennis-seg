DROP TABLE IF EXISTS VT2609246827BE__DWH.s_auth_history CASCADE;

CREATE TABLE VT2609246827BE__DWH.s_auth_history
(
    hk_l_user_group_activity int NOT NULL,
    user_id_from int,
    event varchar(20),
    event_dt timestamp,
    load_dt timestamp,
    load_src varchar(20)
)
PARTITION BY ((s_auth_history.load_dt)::date)
GROUP BY calendar_hierarchy_day((s_auth_history.load_dt)::date, 3, 2);

ALTER TABLE VT2609246827BE__DWH.s_auth_history
    ADD CONSTRAINT fk_s_auth_history_l_user_group_activity
    FOREIGN KEY (hk_l_user_group_activity)
    REFERENCES VT2609246827BE__DWH.l_user_group_activity (hk_l_user_group_activity);

CREATE PROJECTION VT2609246827BE__DWH.s_auth_history
(
    hk_l_user_group_activity,
    user_id_from,
    event,
    event_dt,
    load_dt,
    load_src
)
AS
SELECT
    s_auth_history.hk_l_user_group_activity,
    s_auth_history.user_id_from,
    s_auth_history.event,
    s_auth_history.event_dt,
    s_auth_history.load_dt,
    s_auth_history.load_src
FROM VT2609246827BE__DWH.s_auth_history
ORDER BY s_auth_history.load_dt
SEGMENTED BY s_auth_history.hk_l_user_group_activity ALL NODES;

INSERT INTO VT2609246827BE__DWH.s_auth_history (
    hk_l_user_group_activity,
    user_id_from,
    event,
    event_dt,
    load_dt,
    load_src
)
SELECT
    luga.hk_l_user_group_activity,
    gl.user_id_from,
    gl.event,
    gl.datetime,
    now(),
    's3'
FROM VT2609246827BE__STAGING.group_log AS gl
INNER JOIN VT2609246827BE__DWH.h_users AS hu ON gl.user_id = hu.user_id
INNER JOIN VT2609246827BE__DWH.h_groups AS hg ON gl.group_id = hg.group_id
INNER JOIN VT2609246827BE__DWH.l_user_group_activity AS luga
    ON luga.hk_user_id = hu.hk_user_id
   AND luga.hk_group_id = hg.hk_group_id;
