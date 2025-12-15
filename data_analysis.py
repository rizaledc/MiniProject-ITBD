from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc, count

# 1. Init Spark
print("\n=== TAHAP 2: DATA INTEGRATION & ANALYTICS ===")
spark = SparkSession.builder.appName("ECommerce_Analytics").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 2. Load Data
print("[1] Memuat Data ke Spark DataFrame...")
df_order = spark.read.csv('order_detail.csv', header=True, inferSchema=True)
df_sku = spark.read.csv('sku_detail.csv', header=True, inferSchema=True)
df_pay = spark.read.csv('payment_detail.csv', header=True, inferSchema=True)
df_cust = spark.read.csv('customer_detail_clean.csv', header=True, inferSchema=True)

# 3. Integrasi Data (JOIN)
print("[2] Melakukan Integrasi Data (JOIN Table)...")
# Join: Order -> SKU -> Payment -> Customer
df_joined = df_order.join(df_sku, df_order.sku_id == df_sku.id, "left") \
                    .drop(df_sku.id) \
                    .join(df_pay, df_order.payment_id == df_pay.id, "left") \
                    .drop(df_pay.id) \
                    .withColumnRenamed("payment_method", "payment_type") \
                    .join(df_cust, df_order.customer_id == df_cust.customer_id, "left") \
                    .drop(df_cust.customer_id)

# Hitung Total Profit (Harga Jual - Modal) * Qty
df_joined = df_joined.withColumn("total_profit", (col("price") - col("cogs")) * col("qty_ordered"))

# 4. Analisis Bisnis 1: Profitabilitas per Kategori
print("\n[3] Analisis Profitabilitas per Kategori...")
category_profit = df_joined.groupBy("category") \
                           .agg({"total_profit": "sum", "qty_ordered": "sum"}) \
                           .withColumnRenamed("sum(total_profit)", "total_profit") \
                           .withColumnRenamed("sum(qty_ordered)", "total_sold") \
                           .orderBy(desc("total_profit"))
category_profit.show()

# 5. Analisis Bisnis 2: Metode Pembayaran Terpopuler
print("\n[4] Analisis Metode Pembayaran Terpopuler...")
payment_stats = df_joined.groupBy("payment_type") \
                         .count() \
                         .withColumnRenamed("count", "trx_count") \
                         .orderBy(desc("trx_count"))
payment_stats.show()

# 6. Simpan Dataset Gabungan untuk ML
output_path = "dataset_final_ml"
print(f"\n[5] Menyimpan Dataset Final ke '{output_path}'...")
df_joined.write.mode("overwrite").option("header", "true").csv(output_path)

spark.stop()