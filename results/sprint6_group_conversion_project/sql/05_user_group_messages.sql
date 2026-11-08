WITH user_group_messages AS (
    SELECT
        lgd.hk_group_id,
        COUNT(DISTINCT lum.hk_user_id) AS cnt_users_in_group_with_messages
    FROM VT2609246827BE__DWH.l_groups_dialogs AS lgd
    INNER JOIN VT2609246827BE__DWH.l_user_message AS lum
        ON lgd.hk_message_id = lum.hk_message_id
    GROUP BY lgd.hk_group_id
)
SELECT *
FROM user_group_messages
LIMIT 10;
