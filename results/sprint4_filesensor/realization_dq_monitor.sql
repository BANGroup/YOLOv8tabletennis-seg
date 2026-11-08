Select
	"datetime", 
	table_name,
	dq_check_name,
    dq_check_result
From 
	stage.dq_checks_results
where 
	datetime in (select max(datetime) from stage.dq_checks_results)
	and dq_check_result = 1

-- CREATE SCHEMA stage;
-- create table stage.dq_checks_results
-- (
-- table_name varchar(255),
-- dq_check_name varchar(255),
-- "datetime" TIMESTAMP,
-- dq_check_result numeric(8,2) 
-- )

-- INSERT INTO stage.dq_checks_results (table_name, dq_check_name, "datetime", dq_check_result) VALUES
--     ('test_tn_1', 'test_dq_check_name_1', '2022-01-01 00:00:00', 0),
--     ('test_tn_2', 'test_dq_check_name_2', '2022-01-02 00:00:00', 0),
--     ('test_tn_3', 'test_dq_check_name_3', '2022-01-03 00:00:00', 0),
--     ('test_tn_4', 'test_dq_check_name_4', '2022-01-04 00:00:00', 1),
--     ('test_tn_5', 'test_dq_check_name_5', '2022-01-05 00:00:00', 1);
