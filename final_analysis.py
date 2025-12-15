import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofweek, month, when
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from sklearn.metrics import classification_report, roc_curve, auc

print("=== TAHAP 5: ANALISIS MENDALAM & VISUALISASI ===")
spark = SparkSession.builder.appName("Final_Analysis").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# (Load Data & Feature Engineering sama seperti ml_optimization.py)
# ... [Bagian Load Data disingkat untuk efisiensi laporan, logika sama] ...

# FEATURE IMPORTANCE VISUALIZATION
print("[1] Membuat Grafik Feature Importance...")
# (Asumsi model sudah dilatih dengan parameter terbaik)
rf_optimized = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxDepth=10)
pipeline = Pipeline(stages=[cat_indexer, pay_indexer, assembler, rf_optimized])
model = pipeline.fit(train_data)

# Extract Importances
importances = model.stages[-1].featureImportances
feature_list = ["qty", "price", "disc_ratio", "cat_idx", "pay_idx", "month", "weekend"]

plt.figure(figsize=(10, 6))
plt.barh(feature_list, importances.toArray(), color='teal')
plt.title("Faktor Determinan Validitas Transaksi (Feature Importance)")
plt.savefig('feature_importance_final.png')
print("-> Grafik disimpan: feature_importance_final.png")

# ROC CURVE
print("[2] Membuat Kurva ROC...")
preds = model.transform(test_data)
y_true = preds.select('label').toPandas()
probs = preds.select('probability').toPandas()['probability'].apply(lambda x: x[1])

fpr, tpr, _ = roc_curve(y_true, probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('roc_curve_final.png')
print("-> Grafik disimpan: roc_curve_final.png")

spark.stop()