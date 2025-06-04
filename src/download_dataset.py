import pandas as pd
from pathlib import Path


def download_and_prepare_dataset(url, sample_size, output_path):
    df = pd.read_parquet(url)
    df_sample = df.head(sample_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(output_path, index=False)
    return str(output_path)
    

if __name__ == "__main__":
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
    output_path = Path("data/nyc_taxi_100k.csv")
    sample_size = 100000
    download_and_prepare_dataset(url, sample_size, output_path)
