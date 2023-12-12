# Databricks notebook source
from pyspark.sql.types import IntegerType, DoubleType, BooleanType, DateType
from pyspark.sql.functions import col, lit, format_string, corr, expr, year
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, format_number, round
from pyspark.sql.functions import when

# COMMAND ----------

configs = {"fs.azure.account.auth.type": "OAuth",
"fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
"fs.azure.account.oauth2.client.id": "aa22090c-5056-495d-a551-1df12b469ee5",
"fs.azure.account.oauth2.client.secret": 'k.v8Q~CR~D8DAJ3dtAwBUjXFpzcSsThtKbzQlcar',
"fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/a01ce793-f13f-4f41-80fc-b8847d713420/oauth2/token"}

dbutils.fs.mount(source = "abfss://allriodata@therioproject.dfs.core.windows.net",mount_point = "/mnt/riogames",extra_configs = configs)

# COMMAND ----------

athletes = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/mnt/riogames/raw-data/athletes.csv")
countries = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/mnt/riogames/raw-data/countries.csv")
events = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/mnt/riogames/raw-data/events.csv")

# InferSchema is one way to make sure the columns are the appropriate data type

# COMMAND ----------

athletes.printSchema()

# COMMAND ----------

countries.printSchema()

# COMMAND ----------

events.printSchema()

# COMMAND ----------

#countries = countries.withColumn("population", col("population").cast("Integer"))\
    #.withColumn("gdp_per_capita", col("gdp_per_capita").cast("Integer"))

    # The manual way of changing the data types of the column

# COMMAND ----------

athletes.show()

# COMMAND ----------

medal_dist1 = athletes.drop("id", "sex", "name", "dob", "height", "weight", "sport")   
#Gold, Silver, Bronze Dist. per country

# COMMAND ----------

medal_dist1.show()

# COMMAND ----------

# Define a window specification that orders by total_gold descending, then by total_silver, and then by total_bronze
windowSpec = Window.orderBy(F.col("total_gold").desc(), F.col("total_silver").desc(), F.col("total_bronze").desc())

# COMMAND ----------

medal_dist2 = medal_dist1.groupBy("nationality").agg(
    F.sum("gold").alias("total_gold"),
    F.sum("silver").alias("total_silver"),
    F.sum("bronze").alias("total_bronze")
)\
.join(
    countries, 
    medal_dist1.nationality == countries.code, "inner"
)\
.drop(
    "code",
    "population",
    "gdp_per_capita"
)\
.orderBy(
    F.col("total_gold").desc(),
    F.col("total_silver").desc(),
    F.col("total_bronze").desc()
)\
.withColumn(
    "rank", 
    F.row_number().over(windowSpec)
)\
.select(
    "rank",
    "nationality",
    "country",
    "total_gold",
    "total_silver",
    "total_bronze",
)

# COMMAND ----------

medal_dist2.show()

# COMMAND ----------

medal_standardgdp = medal_dist2.withColumn(
    "TotalMedalCount",col("total_gold") + 
    col("total_silver") + 
    col("total_bronze")
)\
.join(
    countries, 
    medal_dist1.nationality == countries.code, "inner"
)\
.withColumn(
    "adjusted_gdp_per_capita", 
    (col("gdp_per_capita")*col("population"))/(col("population")/lit(1000))
)\
.withColumn(
    "adjusted_gdp_per_capita",
    F.round(col("adjusted_gdp_per_capita"), 3)
)\
.drop(
    "total_gold",
    "total_silver",
    "total_bronze",
    countries["country"],
    "code",
    "adjusted_gdp_per_capita_del",
    "population",
    "gdp_per_capita"
)\
.orderBy(
    "adjusted_gdp_per_capita",
    ascending=False
)\
.dropna(subset=["adjusted_gdp_per_capita"])

# COMMAND ----------

medal_standardgdp.show(n=700)

# COMMAND ----------

correlation_value = medal_standardgdp.stat.corr("TotalMedalCount", "adjusted_gdp_per_capita")
print("Correlation between 'adjusted_gdp_per_capita' and 'TotalMedalCount':", correlation_value)

#0.38 is a weak positive correlation. This means that as the adjusted gdp per capita increases, the total medal count tends to increase as well. In the reverse, as the total medal count increases, the adjusted gdp tends to increase. With the 0.38 correlation, this means that there will be outliers such as Luxembourg. They have a high GDP but won 0 medals in 2016. Similar outliers include: Qatar (1), UAE (1), Finland (1), Iceland (1), etc. Another example is Ethiopia(8)/Kenya (13)/Ukraine(15),etc. they have a low adjusted gdp per capita, but they won a lot of medals in 2016.

#This helps us make the conclusion that while a high GDP can definitely benefit a country's overall medal count, it is not quintessential for a country to have a high gdp per capita to succeed in the Olympics. 

# COMMAND ----------

athletes.show()

# COMMAND ----------

heatmap_height_age_weightM = (
    athletes
    .filter(F.col("sex") == "male")
    .withColumn("TotalMedalCount", 
                F.col("gold") + F.col("silver") + F.col("bronze"))
    .withColumn("CorrectedDob", 
                when(F.year(F.col("dob")) == 2000, F.col("dob"))
                .otherwise(F.expr("add_months(dob, -1200)")))
    .withColumn("age", 
                when(F.year(F.col("dob")) == 2000, 
                     F.expr("2023 - year(dob)"))
                .otherwise(F.expr("2023 - year(CorrectedDob)")))
    .orderBy(F.desc("TotalMedalCount"))
    .select("name", "CorrectedDob", "age", "height", "weight", "TotalMedalCount")
)


# COMMAND ----------

heatmap_height_age_weightM.show()

# COMMAND ----------

correlation_valueMA = heatmap_height_age_weightM.stat.corr("TotalMedalCount", "age")
print(correlation_valueMA)

# COMMAND ----------

correlation_valueMH = heatmap_height_age_weightM.stat.corr("TotalMedalCount", "height")
print(correlation_valueMH)

# COMMAND ----------

correlation_valueMW = heatmap_height_age_weightM.stat.corr("TotalMedalCount", "weight")
print(correlation_valueMW)

# COMMAND ----------

heatmap_height_age_weightF = (
    athletes
    .filter(F.col("sex") == "female")
    .withColumn("TotalMedalCount", 
                F.col("gold") + F.col("silver") + F.col("bronze"))
    .withColumn("corrected_dob", 
                when(F.year(F.col("dob")) == 2000, F.col("dob"))
                .otherwise(F.expr("add_months(dob, -1200)")))
    .withColumn("age", 
                when(F.year(F.col("dob")) == 2000, 
                     F.expr("2023 - year(dob)"))
                .otherwise(F.expr("2023 - year(corrected_dob)")))
    .orderBy(F.desc("TotalMedalCount"))
    .select("name", "corrected_dob", "age", "height", "weight", "TotalMedalCount")
)

# COMMAND ----------

heatmap_height_age_weightF.show()

# COMMAND ----------

correlation_valueFA = heatmap_height_age_weightF.stat.corr("TotalMedalCount", "age")
print(correlation_valueFA)

# COMMAND ----------

correlation_valueFH = heatmap_height_age_weightF.stat.corr("TotalMedalCount", "height")
print(correlation_valueFH)

# COMMAND ----------

correlation_valueFW = heatmap_height_age_weightF.stat.corr("TotalMedalCount", "weight")
print(correlation_valueFW)

# COMMAND ----------

# MAGIC %fs
# MAGIC ls "/mnt/riogames"

# COMMAND ----------

medal_dist2.repartition(1).write.mode("overwrite").option("header","true").csv("/mnt/riogames/transformed-data/TotalMedalsByCountry")
medal_standardgdp.repartition(1).write.mode("overwrite").option("header","true").csv("/mnt/riogames/transformed-data/TotalMedalsByGDP")

# COMMAND ----------

heatmap_height_age_weightM.repartition(1).write.mode("overwrite").option("header","true").csv("/mnt/riogames/transformed-data/TotalMedalsByHWAM") #HWAM= Height, Weight, Age for Males
heatmap_height_age_weightF.repartition(1).write.mode("overwrite").option("header","true").csv("/mnt/riogames/transformed-data/TotalMedalsByHWAF") #HWAF= Height, Weight, Age for Females

# COMMAND ----------

athletes.show()

# COMMAND ----------

#Ideas: 1) By nationality and by sport, discipline who are the highest gold, medal, and silver earners? (only need athletes for this) 2) Which sports and disciplines are most popular across different countries? (athletes and events) 3) How does gender representation vary across different sports and disciplines? (athletes and events)
