from etl_pipeline import run_etl_pipeline
from logging_config import setup_logging


if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Starting NYC Taxi ETL Pipeline")
    
    try:
        run_etl_pipeline()
        logger.info("ETL Pipeline completed successfully")
    except Exception as e:
        logger.error(f"ETL Pipeline failed: {str(e)}", exc_info=True)
        raise
