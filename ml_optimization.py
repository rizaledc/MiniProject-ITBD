from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofweek, month, when
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

print("\n=== TAHAP 4: OPTIMASI MODEL (HYPERPARAMETER TUNING) ===")
spark = SparkSession.builder.appName("ML_Optimization").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load Data Original
df_order = spark.read.csv('order_detail.csv', header=True, inferSchema=True)
df_sku = spark.read.csv('sku_detail.csv', header=True, inferSchema=True)
df_pay = spark.read.csv('payment_detail.csv', header=True, inferSchema=True)

# Join
df = df_order.join(df_sku, df_order.sku_id == df_sku.id, "left") \
             .join(df_pay, df_order.payment_id == df_pay.id, "left") \
             .withColumnRenamed("payment_method", "payment_type")

# FEATURE ENGINEERING LANJUTAN
# Menambah fitur Waktu dan Rasio Diskon
df = df.withColumn("order_month", month("order_date")) \
       .withColumn("order_day", dayofweek("order_date")) \
       .withColumn("is_weekend", when((col("order_day") == 1) | (col("order_day") == 7), 1).otherwise(0)) \
       .withColumn("discount_ratio", when(col("before_discount") > 0, col("discount_amount") / col("before_discount")).otherwise(0)) \
       .withColumn("label", col("is_valid").cast("double"))

# Pipeline Preparation
cat_indexer = StringIndexer(inputCol="category", outputCol="cat_idx", handleInvalid="keep")
pay_indexer = StringIndexer(inputCol="payment_type", outputCol="pay_idx", handleInvalid="keep")

assembler = VectorAssembler(
    inputCols=["qty_ordered", "price", "discount_ratio", "cat_idx", "pay_idx", "order_month", "is_weekend"],
    outputCol="features"
)

rf = RandomForestClassifier(labelCol="label", featuresCol="features")
pipeline = Pipeline(stages=[cat_indexer, pay_indexer, assembler, rf])

# Data Split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# HYPERPARAMETER TUNING
print("[1] Memulai Grid Search & Cross Validation...")
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(metricName="accuracy"),
                          numFolds=3)

cvModel = crossval.fit(train_data)
bestModel = cvModel.bestModel.stages[-1]

# Evaluasi Final
preds = cvModel.transform(test_data)
acc = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(preds)

print(f"\n=== HASIL OPTIMASI ===")
print(f"Akurasi Terbaik: {acc*100:.2f}%")
print(f"Best NumTrees: {bestModel.getNumTrees}")
print(f"Best MaxDepth: {bestModel.getMaxDepth}")

spark.stop()