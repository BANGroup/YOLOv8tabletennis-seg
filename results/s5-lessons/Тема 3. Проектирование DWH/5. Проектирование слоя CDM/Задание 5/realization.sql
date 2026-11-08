-- старое не рабочее
-- CREATE UNIQUE INDEX dm_settlement_report_restaurant_id_settlement_year_settleme_key
--     ON cdm.dm_settlement_report (restaurant_id, settlement_year, settlement_month);

-- старое до 27.06
-- ALTER TABLE cdm.dm_settlement_report
--     ADD CONSTRAINT dm_settlement_report_restaurant_id_settlement_year_settleme_key
--     UNIQUE (restaurant_id, settlement_year, settlement_month);

ALTER TABLE cdm.dm_settlement_report
    ADD CONSTRAINT dm_settlement_report_restaurant_id_settlement_date
    UNIQUE (restaurant_id, settlement_date);    