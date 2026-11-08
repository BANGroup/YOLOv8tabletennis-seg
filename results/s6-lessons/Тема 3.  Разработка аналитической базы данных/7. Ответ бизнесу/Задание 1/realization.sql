SELECT age,
       COUNT(1)
FROM VT2609246827BE__DWH.s_user_socdem
WHERE hk_user_id IN (
    SELECT hk_user_id
    FROM VT2609246827BE__DWH.l_user_message
    WHERE hk_message_id IN (
        SELECT hk_message_id
        FROM VT2609246827BE__DWH.l_groups_dialogs
        WHERE hk_group_id IN (
            SELECT hk_group_id
            FROM VT2609246827BE__DWH.h_groups
            ORDER BY registration_dt
            LIMIT 10
        )
    )
)
GROUP BY age
ORDER BY 2 DESC;
