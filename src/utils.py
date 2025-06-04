from pyspark.sql.functions import *
from pyspark.sql.types import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from logging import getLogger


logger = getLogger(__name__)


def preprocess_data(df):
    logger.info("Starting data preprocessing")
    
    df = df.withColumn("tpep_pickup_datetime", to_timestamp(col("tpep_pickup_datetime"))) \
           .withColumn("tpep_dropoff_datetime", to_timestamp(col("tpep_dropoff_datetime"))) \
           .withColumn("passenger_count", col("passenger_count").cast(IntegerType())) \
           .withColumn("trip_distance", col("trip_distance").cast(FloatType())) \
           .withColumn("fare_amount", col("fare_amount").cast(FloatType())) \
           .withColumn("total_amount", col("total_amount").cast(FloatType())) \
           .withColumn("PULocationID", col("PULocationID").cast(IntegerType())) \
           .withColumn("DOLocationID", col("DOLocationID").cast(IntegerType()))
    
    initial_count = df.count()
    df = df.filter(
        (col("fare_amount") > 0) &
        (col("trip_distance") > 0) &
        (col("passenger_count") > 0) &
        (col("total_amount") > 0) &
        (col("tpep_pickup_datetime") < col("tpep_dropoff_datetime"))
    )
    filtered_count = df.count()
    
    logger.info(f"Filtered out {initial_count - filtered_count} invalid records")
    
    df = df.withColumn(
        "trip_duration_minutes",
        (unix_timestamp(col("tpep_dropoff_datetime")) - unix_timestamp(col("tpep_pickup_datetime"))) / 60
    )
    
    df = df.filter(
        (col("trip_duration_minutes") > 0) &
        (col("trip_duration_minutes") < 180)
    )
    
    logger.info(f"Final clean data count: {df.count()}")
    return df


def train_model(df):
    logger.info("Starting model training")
    
    df['day_of_week'] = df['day'].dt.dayofweek
    df['month'] = df['day'].dt.month
    features = ['PULocationID', 'DOLocationID', 'day_of_week', 'month', 'avg_distance']
    target = 'daily_revenue'
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Train/test split: {len(X_train)}/{len(X_test)} samples")
    
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    logger.info("Fitting RandomForest model...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    
    logger.info(f"Model metrics: {metrics}")
    return model, metrics
