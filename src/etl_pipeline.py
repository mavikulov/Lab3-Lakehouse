import shutil
import os

import mlflow
import pandas as pd
from delta import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    to_timestamp,
    year,
    month,
    date_trunc,
    hour,
    avg,
    sum,
    count
)

from utils import preprocess_data, train_model
from logging import getLogger


logger = getLogger(__name__)


class NYCTaxiETL:
    def __init__(self):
        self.spark = self.init_spark()
        
    def init_spark(self):
        logger.info("Initializing Spark session")
        
        builder = SparkSession.builder \
            .appName("NYCTaxiLakehouse") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \

        logger.info("Initializing Spark session done!")
        
        return configure_spark_with_delta_pip(builder).getOrCreate()
    
    def bronze_layer(self, data_path=None):
        if data_path is None:
            data_path = "/app/data/nyc_taxi_100k.csv"
        
        logger.info(f"Processing Bronze layer from {data_path}")
        
        df = self.spark.read.option("header", "true") \
                            .option("inferSchema", "true") \
                            .csv(data_path)
        
        logger.info(f"Raw data count: {df.count()}")
        
        if "tpep_pickup_datetime" in df.columns:
            df = df.withColumn("tpep_pickup_datetime", to_timestamp(col("tpep_pickup_datetime")))
        
        if "tpep_dropoff_datetime" in df.columns:
            df = df.withColumn("tpep_dropoff_datetime", to_timestamp(col("tpep_dropoff_datetime")))
        
        df = df.withColumn("year", year(col("tpep_pickup_datetime"))) \
            .withColumn("month", month(col("tpep_pickup_datetime")))
        
        df = df.repartition(16, "year", "month")
        
        df.write.format("delta") \
            .partitionBy("year", "month") \
            .mode("overwrite") \
            .save("/app/data/bronze/nyc_taxi_trips")
        
        logger.info("Bronze layer saved successfully")
        return df
    
    def silver_layer(self):
        logger.info("Processing Silver layer")
        
        df = self.spark.read.format("delta").load("/app/data/bronze/nyc_taxi_trips")
        logger.info(f"Bronze data count: {df.count()}")
        
        df_clean = preprocess_data(df)
        logger.info(f"Cleaned data count: {df_clean.count()}")
        
        df_clean.write.format("delta") \
            .mode("overwrite") \
            .save("/app/data/silver/nyc_taxi_trips_clean")
        
        delta_table = DeltaTable.forPath(self.spark, "/app/data/silver/nyc_taxi_trips_clean")
        delta_table.optimize().executeZOrderBy(["PULocationID", "DOLocationID"])
        
        logger.info("Silver layer saved with Z-ordering optimization")
        return df_clean
    
    def gold_layer(self):
        logger.info("Processing Gold layer")
        
        df = self.spark.read.format("delta").load("/app/data/silver/nyc_taxi_trips_clean")
    
        daily_revenue = self._create_daily_revenue_table(df)
        hourly_demand = self._create_hourly_demand_table(df)
        
        logger.info("Gold layer processing completed")
        return daily_revenue, hourly_demand
    
    def _create_daily_revenue_table(self, df):
        logger.info("Creating daily revenue table")
        
        daily_revenue = df.groupBy(
            "PULocationID", 
            "DOLocationID", 
            date_trunc("day", "tpep_pickup_datetime").alias("day")
        ).agg(
            sum("total_amount").alias("daily_revenue"),
            count("*").alias("trip_count"),
            avg("trip_distance").alias("avg_distance"),
            avg("total_amount").alias("avg_fare")
        )
        
        daily_revenue = daily_revenue.repartition(8, "PULocationID")
        
        daily_revenue.write.format("delta") \
            .partitionBy("PULocationID") \
            .mode("overwrite") \
            .save("/app/data/gold/daily_revenue_by_location")
        
        return daily_revenue
    
    def _create_hourly_demand_table(self, df):
        logger.info("Creating hourly demand table")
        
        hourly_demand = df.groupBy(
            "PULocationID",
            hour("tpep_pickup_datetime").alias("hour_of_day")
        ).agg(
            count("*").alias("trip_count"),
            avg("total_amount").alias("avg_fare")
        )
        
        hourly_demand.write.format("delta") \
            .mode("overwrite") \
            .save("/app/data/gold/hourly_demand_patterns")
        
        return hourly_demand
    
    def train_ml_model(self):
        logger.info("Starting ML model training")
        
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("NYC_Taxi_Revenue_Prediction")
        
        df = self.spark.read.format("delta").load("/app/data/gold/daily_revenue_by_location")
        pandas_df = df.toPandas()
        
        with mlflow.start_run() as run:
            logger.info(f"MLflow run started with ID: {run.info.run_id}")
            
            model, metrics = train_model(pandas_df)
            
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", 20)
            mlflow.log_param("random_state", 42)
            
            mlflow.log_metrics(metrics)
            
            input_example = pandas_df.iloc[0:1][['PULocationID', 'DOLocationID', 'day_of_week', 'month', 'avg_distance']]
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="taxi_revenue_model",
                input_example=input_example,
                registered_model_name="TaxiRevenuePredictor"
            )
            
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            feature_importances = pd.Series(model.feature_importances_, index=input_example.columns)
            feature_importances.nlargest(10).plot(kind='barh')
            plt.title("Feature Importances")
            mlflow.log_figure(plt.gcf(), "feature_importances.png")
            plt.close()
            
            logger.info(f"Successfully logged model to MLflow. Run ID: {run.info.run_id}")
            logger.info(f"Model metrics: {metrics}")
            
            model_path = "/app/data/models/taxi_revenue_predictor"
            try:
                shutil.rmtree(model_path)
                logger.info(f"Removed existing model at {model_path}")
            except FileNotFoundError:
                logger.info(f"No existing model found at {model_path}")
            
            os.makedirs(model_path, exist_ok=True)
            mlflow.sklearn.save_model(model, model_path)
            
            return model, metrics


def run_etl_pipeline():
    etl = NYCTaxiETL()
    etl.bronze_layer(data_path=None)
    etl.silver_layer()
    etl.gold_layer()
    etl.train_ml_model()
