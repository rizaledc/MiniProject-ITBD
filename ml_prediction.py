from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# 1. Init Spark
print("\n=== TAHAP 3: MACHINE LEARNING (BASELINE MODEL) ===")
spark = SparkSession.builder.appName("ML_Baseline").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 2. Load Data
data = spark.read.csv('dataset_final_ml', header=True, inferSchema=True)
data = data.withColumn("label", data["is_valid"].cast("double")) # Target Variable

# 3. Preprocessing
cat_indexer = StringIndexer(inputCol="category", outputCol="cat_idx", handleInvalid="keep")
pay_indexer = StringIndexer(inputCol="payment_type", outputCol="pay_idx", handleInvalid="keep")

assembler = VectorAssembler(
    inputCols=["qty_ordered", "price", "cat_idx", "pay_idx"],
    outputCol="features"
)

# 4. Split Data
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# 5. Modeling
# Model A: Logistic Regression
lr = LogisticRegression(labelCol="label", featuresCol="features")
pipeline_lr = Pipeline(stages=[cat_indexer, pay_indexer, assembler, lr])

# Model B: Random Forest (Default)
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=20)
pipeline_rf = Pipeline(stages=[cat_indexer, pay_indexer, assembler, rf])

# 6. Training & Evaluation
print("[1] Melatih Model...")
model_lr = pipeline_lr.fit(train_data)
model_rf = pipeline_rf.fit(train_data)

print("[2] Evaluasi Akurasi...")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

acc_lr = evaluator.evaluate(model_lr.transform(test_data))
acc_rf = evaluator.evaluate(model_rf.transform(test_data))

print(f"-> Akurasi Logistic Regression : {acc_lr*100:.2f}%")
print(f"-> Akurasi Random Forest       : {acc_rf*100:.2f}%")

spark.stop()