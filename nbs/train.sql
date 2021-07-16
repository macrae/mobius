
with
impute_label as (
    select
        * except({{target}}),
        case
            when {{target}} = 1 then 1
            when {{target}} = 0 then 0
            else 0
        end as label
    from transfer_learning.snn_master_table_v1
)
,
ideals as (
    select *
    from impute_label
    where label = 1
)
,
non_ideals as (
    select *
    from impute_label
    where label = 0
)
,
downsample_non_ideals as (
    select *
    from non_ideals
    where True
        and logNetWorth > {{nw_filter}}
        and rand() < {{class_balance}} * {{ideal_size}} / {{non_ideal_size}}
)

select * from ideals
union all
select * from downsample_non_ideals
