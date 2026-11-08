WITH oldest_groups AS (
    SELECT hk_group_id
    FROM VT2609246827BE__DWH.h_groups
    ORDER BY registration_dt
    LIMIT 10
),
user_group_messages AS (
    SELECT
        lgd.hk_group_id,
        COUNT(DISTINCT lum.hk_user_id) AS cnt_users_in_group_with_messages
    FROM VT2609246827BE__DWH.l_groups_dialogs AS lgd
    INNER JOIN VT2609246827BE__DWH.l_user_message AS lum
        ON lgd.hk_message_id = lum.hk_message_id
    GROUP BY lgd.hk_group_id
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
SELECT
    ugl.hk_group_id,
    ugl.cnt_added_users,
    COALESCE(ugm.cnt_users_in_group_with_messages, 0) AS cnt_users_in_group_with_messages,
    COALESCE(ugm.cnt_users_in_group_with_messages, 0) / ugl.cnt_added_users AS group_conversion
FROM user_group_log AS ugl
LEFT JOIN user_group_messages AS ugm ON ugl.hk_group_id = ugm.hk_group_id
ORDER BY group_conversion DESC;
