(SELECT COUNT(1) AS total,
        COUNT(DISTINCT u.id) AS uniq,
        'users' AS dataset
FROM VT2609246827BE__STAGING.users u)
UNION ALL
(SELECT COUNT(1),
        count(DISTINCT g.id),
        'groups'
FROM VT2609246827BE__STAGING.groups g)
UNION ALL
(SELECT COUNT(1),
        COUNT(DISTINCT d.message_id),
        'dialogs'
FROM VT2609246827BE__STAGING.dialogs d);
