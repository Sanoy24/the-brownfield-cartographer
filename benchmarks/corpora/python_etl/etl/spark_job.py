def build_customer_totals(spark):
    spark.sql(
        "CREATE OR REPLACE TABLE marts.customer_totals AS "
        "SELECT customer_id, sum(amount) AS total FROM staging.orders GROUP BY customer_id"
    )
    spark.table("marts.customer_totals").write.parquet("exports/customer_totals")


def backfill(spark, day):
    spark.sql(f"INSERT INTO marts.daily SELECT * FROM staging.events WHERE day = '{day}'")
