create table dds.dm_timestamps
(
    id    serial primary key,
    ts    timestamp not null,
    year  integer   not null
        constraint dm_timestamps_year_check
            check ((year >= 2022) AND (year < 2500)),
    month integer   not null
        constraint dm_timestamps_month_check
            check ((month >= 1) AND (month <= 12)),
    day   integer   not null
        constraint dm_timestamps_day_check
            check ((day >= 1) AND (day <= 31)),
    time  time      not null,
    date  date      not null
);