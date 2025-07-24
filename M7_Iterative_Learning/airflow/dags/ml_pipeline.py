from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os

def check_drift():
    os.system("python monitor.py")
    # Optionally: check drift_report.html or drift_report.json for drift and notify
    # (for demo, always print)
    os.system("python notify.py 'Drift detection run complete'")

def retrain():
    os.system("python retrain.py")

default_args = {'owner': 'airflow', 'retries': 0, 'retry_delay': timedelta(minutes=5)}
dag = DAG('ml_iterative_pipeline', default_args=default_args,
          start_date=datetime(2024, 1, 1), schedule_interval='@daily', catchup=False)

check_drift_task = PythonOperator(task_id='check_drift', python_callable=check_drift, dag=dag)
retrain_task = PythonOperator(task_id='retrain', python_callable=retrain, dag=dag)

check_drift_task >> retrain_task
