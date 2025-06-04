#!/bin/bash

rm -rf /app/data/models/*
mkdir -p /app/data/models
mkdir -p /app/data/bronze /app/data/silver /app/data/gold 

mlflow ui --host 0.0.0.0 --port 5000 &

python /app/src/download_dataset.py
python /app/src/main.py

tail -f /dev/null