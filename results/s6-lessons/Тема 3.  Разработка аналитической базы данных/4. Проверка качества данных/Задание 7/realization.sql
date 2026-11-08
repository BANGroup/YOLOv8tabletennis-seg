(SELECT
    count(1),
    'missing group admin info' as info
FROM VT2609246827BE__STAGING.groups g
LEFT JOIN VT2609246827BE__STAGING.users u
ON g.admin_id = u.id
WHERE u.id is NULL)
UNION ALL
(SELECT
    count(1),
    'missing sender info'
FROM VT2609246827BE__STAGING.dialogs d
LEFT JOIN VT2609246827BE__STAGING.users u
ON d.message_from = u.id
WHERE u.id is NULL)
UNION ALL
(SELECT
    count(1),
    'missing receiver info'
FROM VT2609246827BE__STAGING.dialogs d
LEFT JOIN VT2609246827BE__STAGING.users u
ON d.message_to = u.id
WHERE u.id is NULL)
UNION ALL
(SELECT
    count(1),
    'norm receiver info'
FROM VT2609246827BE__STAGING.dialogs d
LEFT JOIN VT2609246827BE__STAGING.users u
ON d.message_to = u.id
WHERE u.id is NOT NULL)
;
