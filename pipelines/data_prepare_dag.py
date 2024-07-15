import os
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor

# take todays datetime - 10 minutes
start_date = pendulum.now(tz="Europe/Moscow").subtract(minutes=10)
start_date = start_date.replace(second=0, microsecond=0)
project_root = os.environ.get("PROJECT_DIR")


with DAG(
    dag_id="data_prepare_dag",
    description="A data preparation DAG that depends on data extraction DAG",
    schedule_interval=timedelta(minutes=5),
    start_date=start_date,
    tags=["data preparation"],
    # catchup=False,
    is_paused_upon_creation=False,
) as data_prepare_dag:

    # Task to wait for the completion of the data_extract_dag
    wait_for_data_extraction = ExternalTaskSensor(
        task_id="wait_for_data_extraction",
        external_dag_id="data_extract_dag",
        # external_task_id=None,  # Wait for the whole DAG to complete
        # timeout=150,
        # check_existence=True,
        # mode="reschedule"
    )

    run_zenml_pipeline = BashOperator(
        task_id="run_zenml_pipeline",
        bash_command="python3 pipelines/data_prepare.py -prepare_data_pipeline",
        cwd=project_root,  # specifies the current working directory
    )

    wait_for_data_extraction >> run_zenml_pipeline
