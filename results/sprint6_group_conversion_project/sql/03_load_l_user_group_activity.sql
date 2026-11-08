INSERT INTO VT2609246827BE__DWH.l_user_group_activity (
    hk_l_user_group_activity,
    hk_user_id,
    hk_group_id,
    load_dt,
    load_src
)
SELECT DISTINCT
    hash(hu.hk_user_id, hg.hk_group_id),
    hu.hk_user_id,
    hg.hk_group_id,
    now(),
    's3'
FROM VT2609246827BE__STAGING.group_log AS gl
LEFT JOIN VT2609246827BE__DWH.h_users AS hu ON gl.user_id = hu.user_id
LEFT JOIN VT2609246827BE__DWH.h_groups AS hg ON gl.group_id = hg.group_id
WHERE hash(hu.hk_user_id, hg.hk_group_id) NOT IN (
    SELECT hk_l_user_group_activity
    FROM VT2609246827BE__DWH.l_user_group_activity
)
  AND hu.hk_user_id IS NOT NULL
  AND hg.hk_group_id IS NOT NULL;
