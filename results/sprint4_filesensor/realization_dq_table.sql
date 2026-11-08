create table stage.dq_checks_results
(
table_name varchar(255),
dq_check_name varchar(255),
"datetime" TIMESTAMP,
dq_check_result numeric(8,2) 
)

