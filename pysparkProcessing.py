from pyspark.sql import functions as F
from pyspark.sql.functions import radians, sin, cos, atan2, sqrt
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import os
import shutil

def process_transaction_data(input_file_path, output_file_name):
    """Process transaction data and save as processed CSV"""
    
    # Set up Spark environment
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
    os.environ['HADOOP_HOME'] = r"C:\hadoop"
    os.environ["PYSPARK_PYTHON"] = r"C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe"
    os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe"

    # Initializing Spark session  
    spark = SparkSession.builder \
        .appName("TruLedger") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

    try:
        # Load dataset
        print(f"📂 Loading dataset: {input_file_path}")
        df = spark.read.csv(input_file_path, header=True, inferSchema=True)

        # =============================================
        # 🔍 FEATURE ENGINEERING & DATA PREPROCESSING
        # =============================================

        # DROPPED COLUMNS THAT ARE NOT USEFUL FOR ANOMALY DETECTION
        df = df.drop("index", "merchant", "first", "last", "gender", "street", "city", "zip", 
                      "city_pop", "trans_num", "unix_time")

        # EXTRACTING TIME FROM DATE_TIME COLUMN
        df = df.withColumn("trans_date_trans_time", F.date_format(F.to_timestamp("trans_date_trans_time"), "HH").cast("int"))
        df = df.withColumnRenamed("trans_date_trans_time", "txn_time")

        # MERCH/TXN CATEGORIES
        df = df.withColumn("category", F.regexp_replace(F.col("category"), ",", " -"))
        Categories_list = [ctg['category'] for ctg in df.select("category").distinct().collect()]
        for category in Categories_list:
            df = df.withColumn(f"TXNctg_{category}", F.when(F.col("category") == category, 1).otherwise(0))
        df = df.drop("category")

        # STATE COLUMNS
        States_list = [st['state'] for st in df.select("state").distinct().collect()]
        for state in States_list:
            df = df.withColumn(f"state_{state}", F.when(F.col("state") == state, 1).otherwise(0))
        df = df.drop("state")

        # JOB COLUMNS
        jobcat_df = spark.read.csv("c:/Users/hp/LNU/TruLedger-AI/Datasets/Processed/job_categories.csv", header=True, inferSchema=True)
        df = df.withColumn("job", F.regexp_replace(F.col("job"), ",", " -"))

        job_category_pairs = []
        for category in jobcat_df.columns:
            jobs_in_cat = jobcat_df.select(category).where(F.col(category).isNotNull()).distinct().rdd.flatMap(lambda x: x).collect()
            job_category_pairs.extend([(job, category) for job in jobs_in_cat])

        job_category_map_df = spark.createDataFrame(job_category_pairs, ["job", "job_category"])
        df = df.join(job_category_map_df, on="job", how="left")

        unique_job_categories = [row["job_category"] for row in job_category_map_df.select("job_category").distinct().collect()]
        for category in unique_job_categories:
            df = df.withColumn(
                f"JOBctg_{category.replace(' ', '_')}",
                F.when(F.col("job_category") == category, 1).otherwise(0)
            )
        df = df.drop("job_category")
        df = df.drop("job")

        # DOB COLUMNS
        df = df.withColumn("dob_year", F.year(F.to_date("dob", "yyyy-MM-dd")))
        decades = list(range(1920, 2010, 10))
        for start_year in decades:
            col_name = f"dob_{str(start_year)[2:]}s"
            df = df.withColumn(
                col_name,
                F.when((F.col("dob_year") >= start_year) & (F.col("dob_year") < start_year + 10), 1).otherwise(0)
            )
        df = df.drop("dob_year")
        df = df.drop("dob")

        # AVERAGE DISTANCE BETWEEN CUSTOMER AND MERCHANT PER USER
        df = df.withColumn("lat1_rad", radians("lat"))
        df = df.withColumn("lon1_rad", radians("long"))  
        df = df.withColumn("lat2_rad", radians("merch_lat"))
        df = df.withColumn("lon2_rad", radians("merch_long"))

        df = df.withColumn("dlat", F.col("lat2_rad") - F.col("lat1_rad"))
        df = df.withColumn("dlon", F.col("lon2_rad") - F.col("lon1_rad"))

        df = df.withColumn("a", 
            sin(F.col("dlat") / 2) ** 2 + 
            cos(F.col("lat1_rad")) * cos(F.col("lat2_rad")) * 
            sin(F.col("dlon") / 2) ** 2
        )
        df = df.withColumn("c", 2 * atan2(sqrt(F.col("a")), sqrt(1 - F.col("a"))))
        df = df.withColumn("distance", F.col("c") * 6371)

        # USER BEHAVIORAL METRICS
        user_metrics_df = df.groupBy("cc_num").agg(
            F.round(F.avg("amt"), 2).alias("avg_txn_amt"),
            F.round(F.stddev("amt"), 2).alias("stddev_txn_amt"),
            F.round(F.avg("txn_time"), 2).alias("avg_txn_time"),
            F.round(F.avg("distance"), 2).alias("avg_merchant_distance")
        )
        df = df.join(user_metrics_df, on="cc_num", how="left")

        # Dropping temporary columns
        df = df.drop("lat1_rad", "lon1_rad", "lat2_rad", "lon2_rad", "dlat", "dlon", "a", "c", "distance")
        df = df.drop("cc_num", "lat", "long", "merch_lat", "merch_long")

        # Reordering columns to have 'is_fraud' as the first column
        other_columns = [col for col in df.columns if col != "is_fraud"]
        new_column_order = ["is_fraud"] + other_columns
        df = df.select(*new_column_order)

        # Function to clean up Spark CSV output
        def clean_spark_output(temp_dir, final_file_path):
            for file in os.listdir(temp_dir):
                if file.startswith("part-") and file.endswith(".csv"):
                    shutil.move(os.path.join(temp_dir, file), final_file_path)
                    break
            shutil.rmtree(temp_dir)

        # Write processed data
        temp_dir = "c:/Users/hp/LNU/TruLedger-AI/temp_output"
        output_dir = "c:/Users/hp/LNU/TruLedger-AI/Uploaded_Datasets/Processed"
        final_file_path = os.path.join(output_dir, output_file_name)
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        df.coalesce(1).write.csv(temp_dir, header=True, mode="overwrite")
        clean_spark_output(temp_dir, final_file_path)
        
        print(f"✅ Successfully processed and saved: {final_file_path}")
        return final_file_path
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return None
    finally:
        spark.stop()

# For standalone testing
if __name__ == "__main__":
    # Test with one of your files
    test_file = "c:/Users/hp/LNU/TruLedger-AI/TransactionLogs-1.csv"
    output_name = "ProcessedTransactionLogs-1.csv"
    process_transaction_data(test_file, output_name)