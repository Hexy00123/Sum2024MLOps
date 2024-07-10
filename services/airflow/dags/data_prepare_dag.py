from datetime import timedelta

from airflow import DAG
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
        'data_prepare_dag',
        default_args=default_args,
        description='A data preparation DAG that depends on data extraction DAG',
        schedule_interval=timedelta(minutes=5),
        start_date=days_ago(1),
        tags=['example'],
) as dag:
    # Task to wait for the completion of the data_extract_dag
    wait_for_data_extraction = ExternalTaskSensor(
        task_id='wait_for_data_extraction',
        external_dag_id='hello_dag',
        external_task_id=None,  # Wait for the whole DAG to complete
        timeout=600,
    )

    run_zenml_pipeline = BashOperator(
        task_id="run_zenml_pipeline",
        bash_command="python3 pipelines/data_prepare.py -prepare_data_pipeline",
        cwd="/mnt/c/Users/sasha/PycharmProjects/Sum2024MLOps",  # specifies the current working directory
    )

    # Define the task dependencies
    wait_for_data_extraction >> run_zenml_pipeline
