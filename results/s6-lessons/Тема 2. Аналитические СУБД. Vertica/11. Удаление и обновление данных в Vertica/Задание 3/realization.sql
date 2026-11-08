SET ROLE VT2609246827BE_A;

DROP TABLE IF EXISTS members_inc CASCADE;
DROP TABLE IF EXISTS members_rej CASCADE;
DROP TABLE IF EXISTS members CASCADE;

CREATE TABLE members
(
    id int NOT NULL,
    age int,
    gender char,
    email varchar(50),
    CONSTRAINT C_PRIMARY PRIMARY KEY (id) DISABLED
)
ORDER BY id
SEGMENTED BY HASH(id) ALL NODES;

COPY members(id, age, gender, email ENFORCELENGTH)
FROM LOCAL '/path/to/members.csv'
DELIMITER ';'
REJECTED DATA AS TABLE members_rej;

CREATE TABLE members_inc
(
    id int NOT NULL,
    age int,
    gender char,
    email varchar(50),
    CONSTRAINT C_PRIMARY PRIMARY KEY (id) DISABLED
)
ORDER BY id
SEGMENTED BY HASH(id) ALL NODES;

COPY members_inc(id, age, gender, email ENFORCELENGTH)
FROM LOCAL '/path/to/members_inc.csv'
DELIMITER ';'
SKIP 1
REJECTED DATA AS TABLE members_rej;

MERGE INTO members tgt
USING members_inc src
ON tgt.id = src.id
WHEN MATCHED AND NOT (
    tgt.age <=> src.age AND tgt.gender <=> src.gender AND tgt.email <=> src.email
)
THEN UPDATE SET gender = src.gender, email = src.email, age = src.age
WHEN NOT MATCHED THEN INSERT (id, age, gender, email)
VALUES (src.id, src.age, src.gender, src.email);
