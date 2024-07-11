from airflow import DAG
from airflow.decorators import task
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago
from datetime import timedelta

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
        external_dag_id='data_extract_dag',
        external_task_id=None,  # Wait for the whole DAG to complete
        timeout=600,  # Timeout after 10 minutes
        allowed_states=['success'],  # Only proceed if the DAG is successful
        failed_states=['failed', 'skipped'],  # Fail if the DAG failed or was skipped
        mode='poke',  # Check the status of the external task at regular intervals
    )

    @task.bash(task_id="run_zenml_pipeline", cwd="/mnt/c/Users/sasha/PycharmProjects/Sum2024MLOps")
    def run_zenml_pipeline():
        return "python3 pipelines/data_prepare.py"

    # Define the task dependencies
    wait_for_data_extraction >> run_zenml_pipeline()
