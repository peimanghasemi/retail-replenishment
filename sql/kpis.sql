-- Availability, stockouts, write-offs and stock cover per group.
-- {group} is replaced by the grouping columns, e.g. "policy" or "policy, category".
SELECT
    {group},
    SUM(sold) / NULLIF(SUM(demand), 0)                      AS fill_rate,
    SUM(stockout_days) * 1.0 / NULLIF(SUM(open_days), 0)    AS stockout_rate,
    SUM(written_off) / NULLIF(SUM(received), 0)             AS writeoff_rate,
    SUM(inventory_sum) / NULLIF(SUM(demand), 0)             AS days_of_stock
FROM sim
GROUP BY {group}
ORDER BY {group};
