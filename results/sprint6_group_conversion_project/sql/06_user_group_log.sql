WITH oldest_groups AS (
    SELECT hk_group_id
    FROM VT2609246827BE__DWH.h_groups
    ORDER BY registration_dt
    LIMIT 10
),
user_group_log AS (
    SELECT
        luga.hk_group_id,
        COUNT(DISTINCT luga.hk_user_id) AS cnt_added_users
    FROM VT2609246827BE__DWH.l_user_group_activity AS luga
    INNER JOIN VT2609246827BE__DWH.s_auth_history AS sah
        ON luga.hk_l_user_group_activity = sah.hk_l_user_group_activity
    WHERE sah.event = 'add'
      AND luga.hk_group_id IN (SELECT hk_group_id FROM oldest_groups)
    GROUP BY luga.hk_group_id
)
SELECT *
FROM user_group_log;
