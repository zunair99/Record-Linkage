from tokenize import Double
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, stddev, expr
from pyspark.sql.types import DoubleType
from pyspark.sql import DataFrame

spark = SparkSession.Builder().appName("Record Linkage").getOrCreate()

df = spark.read.option("nullValue","?").csv("linkage/block*.csv", header=True,inferSchema=True)
df.show(2)
df.printSchema()
df.cache()

#Grouping count of records by Patient Match
df.groupBy("is_match").count().orderBy(col("count").desc()).show()

#mean and standard deviation for cmp_sex
df.agg(avg("cmp_sex"), stddev("cmp_sex")).show()

#create temporary view
df.createOrReplaceTempView("rec_linkage")
spark.sql(
    """
    select
        is_match,
        count(*) as cnt
    from rec_linkage
    group by 1
    order by 2 desc
    """
).show()

df_summary = df.describe()
df_summary.show()
df_summary.select("summary","cmp_fname_c1","cmp_fname_c2").show()

matched = df.filter("is_match = true")
matched_summary = matched.describe()
matched_summary.show()

missed = df.filter("is_match = false")
missed_summary = missed.describe()
missed_summary.show()

#convert Spark DF into Pandas DF to transpose the data
pd_summary = df_summary.toPandas()
pd_summary.head()
pd_summary.shape

#Transpose Pandas DF
pd_summary = pd_summary.set_index("summary").transpose().reset_index()
pd_summary = pd_summary.rename(columns = {'index':'field'})
pd_summary = pd_summary.rename_axis(None, axis = 1)
pd_summary.shape

#Convert transposed Pandas DF back into Spark DF
summary_df = spark.createDataFrame(pd_summary)
summary_df.show()
summary_df.printSchema() #all records treated as string

#Convert values from string to double
for c in summary_df.columns:
    if c == 'field':
        continue
    summary_df = summary_df.withColumn(c, summary_df[c].cast(DoubleType()))
summary_df.printSchema() #all converted to double except field column

#create function to transpose summary df for matched and missed dataframes
def pivot_sum(desc):
    #convert to PD Dataframe
    pd_desc = desc.toPandas()

    #transpose PD Dataframe
    pd_desc = pd_desc.set_index('summary').transpose().reset_index()
    pd_desc = pd_desc.rename(columns={'index':'field'})
    pd_desc = pd_desc.rename_axis(None, axis=1)

    #convert to Spark Dataframe
    df_desc = spark.createDataFrame(pd_desc)

    #convert metrics from string to double
    for c in df_desc.columns:
        if c == 'field':
            continue
        df_desc = df_desc.withColumn(c, df_desc[c].cast(DoubleType()))
    return df_desc

#transposing matched and missed summary dataframes
matched_summary_df = pivot_sum(matched_summary)
missed_summary_df = pivot_sum(missed_summary)

#joining matched and missed dataframes
matched_summary_df.createOrReplaceTempView("matched_sum")
missed_summary_df.createOrReplaceTempView("missed_sum")
spark.sql(
    """
    select
        a.field,
        (a.count + b.count) as total,
        (a.mean - b.mean) as delta
    from matched_sum a
        inner join missed_sum b on a.field = b.field
    where a.field not in ("id_1","id_2")
    order by delta desc, total desc
    """
).show()

#scoring function
    #sum up values of most significant fields (
    # cmp_lname_c1,
    # cmp_plz,
    # cmp_by,
    # cmp_bd,
    # cmp_bm)
        #these values contain the largest delta and largest total values (relatively)

feats = [
    "cmp_lname_c1",
    "cmp_plz",
    "cmp_by",
    "cmp_bd",
    "cmp_bm"
]
sum_expr = " + ".join(feats)

#using sum_expr to calculate the score
    #replace nulls with 0

scored = df.fillna(0,subset=feats).withColumn('score',expr(sum_expr)).select('score','is_match')
scored.show()

#creating a contingency table to count the number of records whose scores fall above or below a specific threshold value
def cross_tabs(scored: DataFrame, t: DoubleType) -> DataFrame:
    return scored.selectExpr(f"score >= {t} as above", "is_match").groupBy("above").pivot("is_match",("true","false")).count()

cross_tabs(scored, 4.0).show()