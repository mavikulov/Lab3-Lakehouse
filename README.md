# Lab3-Lakehouse
## Данные 

Взят следующий датасет: [NYC Yellow Taxi Trip Data](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data).   Набор данных обрезан до 100000 сэмплов.  

**Цель** — предсказать дневную выручку такси(*daily_revenue*) за день на основе погодных, временных и поведенческих признаков.

## Модель
Использовался `sklearn.ensemble.RandomForestRegressor` случайный лес для классификации. Код обучения приведен в `utils.py`.

## Структура 
```										
├── 📂 data
│	├── 📊 nyc_taxi_100k.csv
├── 📂 src	
│	├── 🐍 download.py
│	├── 🐍 etl_pipeline.py
│	├── 🐍 logging_config.py
│	├──	🐍 main.py				
│   └── 🐍 utils.py	
├── 🐳 Dockerfile				
├──	🐙 docker-compose.yml	
├── ⚙️ start.sh											
└── 📝 README.md				
```

## Запуск
Запустить проект можно с помощью одной команды:
```
docker-compose up -d
```
Опционально можно отдельно запустить скрипт ```download_dataset.py```, который "обрезает" датасет до 100000 записей. Запуск `main.py` выполняется автоматически за счет bash-скрипта `start.sh`.  
Далее выполняется следующий пайплайн:  
[CSV Data] → [Bronze (Raw Data)] → [Silver (Cleaned Data)] → [Gold (Aggregated Data)] → [ML Model Predcition]  

Исполнение каждого этапа логируется в `logs/app.log` в контейнере или в `./logs` в проекте (появляется при запуске контейнера).



## Результаты
Результат запуска представлен в логах `logs/app.log` :
Помимо этого по адресу ``http://localhost:5000/`` можно посмотреть на запуски, модели и метрики.
