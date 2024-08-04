# Databricks notebook source
# MAGIC %md
# MAGIC # Raw to Bronze
# MAGIC
# MAGIC This notebook performs exploratory data analysis on the raw data of CitiBike data. It then saves data in a bronze table after doing priliminary cleaning and processing.

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", "4")

# COMMAND ----------

# MAGIC %md
# MAGIC Let's first look at some rows of the raw table.

# COMMAND ----------

citibike_raw_df = spark.table("citibike_raw")
display(citibike_raw_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dropping Columns
# MAGIC
# MAGIC I don't need station id and names for data driven demand prediction task. So, I'll be removing these columns from the dataset.

# COMMAND ----------

filtered_df = citibike_raw_df.select(
    [
        col
        for col in citibike_raw_df.columns
        if col
        not in [
            "start_station_id",
            "start_station_name",
            "end_station_id",
            "end_station_name",
            "bikeid",
        ]
    ]
)

display(filtered_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imputation
# MAGIC
# MAGIC Filling in missing values for different columns.

# COMMAND ----------

from pyspark.sql.functions import col, isnan, count, when

def get_null_count(df, column):
    """Counts the number of null values in `column`"""

    null_birth_year_count = df.select(
        count(col(column)).alias("non_null_count")
    ).collect()[0][0]
    total_count = df.count()
    null_count = total_count - null_birth_year_count
    return null_count

# COMMAND ----------

# MAGIC %md
# MAGIC ### **usertype** and **gender**

# COMMAND ----------

print("Number of null values in usertype:", get_null_count(filtered_df, "usertype"))
print("Number of null values in gender:", get_null_count(filtered_df, "gender"))

# COMMAND ----------

# MAGIC %md
# MAGIC Since there are no missing values for **gender**, I'll use it to fill in the null values in **usertype** column. I'll fill in using groupby mode.

# COMMAND ----------

from pyspark.sql.functions import col, count, first, expr

# Calculate the mode of usertype for each gender group
mode_usertype_df = (
    filtered_df.groupBy("gender", "usertype")
    .agg(count("usertype").alias("count"))
    .orderBy(col("count").desc())
)

# Get the first usertype (mode) for each gender group
mode_usertype_df = mode_usertype_df.groupBy("gender").agg(
    first("usertype").alias("mode_usertype")
)

# Join the mode_usertype_df with the original filtered_df to get the mode values
joined_df = filtered_df.join(mode_usertype_df, on="gender", how="left")

# Fill null values in usertype with the corresponding mode value
filled_df = joined_df.withColumn(
    "usertype", expr("coalesce(usertype, mode_usertype)")
).drop("mode_usertype")

display(filled_df)

# COMMAND ----------

# MAGIC %md
# MAGIC After imputation, let's check how many null values there are in **usertype** column.

# COMMAND ----------

print("Number of null values in usertype:", get_null_count(filled_df, "usertype"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### birth_year

# COMMAND ----------

print("Number of null values in birth_year:", get_null_count(filled_df, "birth_year"))

# COMMAND ----------

from pyspark.sql.functions import expr, percentile_approx

# Calculate the median birth_year for each usertype and gender
median_birth_year_df = filled_df.groupBy("usertype", "gender").agg(
    percentile_approx("birth_year", 0.5).alias("median_birth_year")
)

# Join the median_birth_year_df with the original filled_df to get the median values
joined_df = filled_df.join(
    median_birth_year_df, on=["usertype", "gender"], how="left"
)

# Fill null values in birth_year with the corresponding median value
filled_df = joined_df.withColumn(
    "birth_year", expr("coalesce(birth_year, median_birth_year)")
).drop("median_birth_year")

display(filled_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Sanity checking after imputing values.

# COMMAND ----------

print("Number of null values in birth_year:", get_null_count(filled_df, "birth_year"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Latitude and Longitude

# COMMAND ----------

from pyspark.sql.functions import col, sum

# Count the number of null values in start and end station latitude and longitude
null_counts_df = filled_df.select(
    sum(col("start_station_latitude").isNull().cast("int")).alias("null_start_station_latitude"),
    sum(col("start_station_longitude").isNull().cast("int")).alias("null_start_station_longitude"),
    sum(col("end_station_latitude").isNull().cast("int")).alias("null_end_station_latitude"),
    sum(col("end_station_longitude").isNull().cast("int")).alias("null_end_station_longitude")
)

display(null_counts_df)

# COMMAND ----------

# MAGIC %md
# MAGIC These values can't be imputed. So I'm deleting them.

# COMMAND ----------

initial_row_count = filled_df.count()

cleaned_df = filled_df.filter(
    (
        col("end_station_latitude").isNotNull() & 
        col("end_station_longitude").isNotNull()
    )
)

final_row_count = cleaned_df.count()

print(f"Number of rows before dropping: {initial_row_count}")
print(f"Number of rows after dropping: {final_row_count}")
print(f"Number of dropped rows: {initial_row_count - final_row_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleaning Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### birth_year

# COMMAND ----------

value_count_df = (
    cleaned_df
    .select("birth_year")
    .groupby("birth_year")
    .count()
    .orderBy("birth_year")
)

value_count_df = value_count_df.toPandas()

# COMMAND ----------

import matplotlib.pyplot as plt

# Create a histogram from value_count_df
plt.figure(figsize=(10, 6))
plt.bar(value_count_df['birth_year'], value_count_df['count'], color='blue')
plt.xlabel('Birth Year')
plt.ylabel('Count')
plt.title('Histogram of Birth Year Counts')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC There seems to be many values that are before 1920s. These values are mistakes in the data.

# COMMAND ----------

count_before_1920 = cleaned_df.filter(cleaned_df.birth_year < 1920).count()
count_before_1920

# COMMAND ----------

# MAGIC %md
# MAGIC These anomalies should be removed. 

# COMMAND ----------

cleaned_df = cleaned_df.filter(cleaned_df.birth_year >= 1920)
display(cleaned_df)

# COMMAND ----------

value_count_df = (
    cleaned_df
    .select("birth_year")
    .groupby("birth_year")
    .count()
    .orderBy("birth_year")
)

value_count_df = value_count_df.toPandas()

# Create a histogram from value_count_df
plt.figure(figsize=(10, 6))
plt.bar(value_count_df['birth_year'], value_count_df['count'], color='blue')
plt.xlabel('Birth Year')
plt.ylabel('Count')
plt.title('Histogram of Birth Year Counts')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

cleaned_df.write.saveAsTable("citibike_bronze", format="delta", mode="overwrite")
