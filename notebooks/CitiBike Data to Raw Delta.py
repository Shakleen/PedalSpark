# Databricks notebook source
import pyspark
from pyspark.sql.types import (
    StructField,
    StructType,
    IntegerType,
    TimestampType,
    StringType,
    FloatType,
)
from pyspark.sql.functions import col

# COMMAND ----------

schema = StructType(
    [
        StructField("tripduration", IntegerType(), False),
        StructField("starttime", TimestampType(), False),
        StructField("stoptime", TimestampType(), False),
        StructField("start station id", IntegerType(), False),
        StructField("start station name", StringType(), False),
        StructField("start station latitude", FloatType(), False),
        StructField("start station longitude", FloatType(), False),
        StructField("end station id", IntegerType(), False),
        StructField("end station name", StringType(), False),
        StructField("end station latitude", FloatType(), False),
        StructField("end station longitude", FloatType(), False),
        StructField("bikeid", IntegerType(), True),
        StructField("usertype", StringType(), True),
        StructField("birth year", IntegerType(), True),
        StructField("gender", IntegerType(), True),
    ]
)

# COMMAND ----------

df = spark.read.csv("/Volumes/pedalsparkws/default/citibank", schema=schema, header=True)

# COMMAND ----------

display(df)

# COMMAND ----------

df.columns

# COMMAND ----------

df = df.select([col(x).alias(x.replace(" ", "_")) for x in df.columns])
df.columns

# COMMAND ----------

df.write.saveAsTable("citibike_raw", format="delta", mode="overwrite")
