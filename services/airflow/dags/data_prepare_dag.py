from datetime import timedelta
import os


from airflow import DAG
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.operators.bash_operator import BashOperator

from airflow.decorators import task
import pendulum


# take todays datetime - 10 minutes
start_date = pendulum.now(tz="Europe/Moscow").subtract(minutes=10)
start_date = start_date.replace(second=0, microsecond=0)


# from data_extract_dag import data_extract_dag
# from hello_dag import hello_dag
# default_args = {
#     'owner': 'airflow',
#     'depends_on_past': False,
#     'email_on_failure': False,
#     'email_on_retry': False,
#     'retries': 1,
#     'retry_delay': timedelta(minutes=2),
# }


# with DAG(dag_id="hello_world",
#          start_date=days_ago(0),
#          schedule_interval=timedelta(minutes=2),
        
#         #  schedule_interval=timedelta(minutes=2)
# ) as hello_dag:
#     # Tasks are represented as operators
#     # Use Bash operator to create a Bash task
#     hello = BashOperator(task_id="hello", bash_command="echo hello")


#     # Python task
#     @task()
#     def world():
#         print("world")


#     # Set dependencies between tasks
#     # First is hello task then world task
#     hello >> world()

with DAG(
        dag_id='data_prepare_dag',
        # default_args=default_args,
        description='A data preparation DAG that depends on data extraction DAG',
        schedule_interval=timedelta(minutes=5),
        # schedule_interval=None,
        start_date=start_date,
        tags=['data preparation'],
        # catchup=False,
        is_paused_upon_creation=False,
) as data_prepare_dag:
    project_root = 'home/sshk/project'
    # Task to wait for the completion of the data_extract_dag
    wait_for_data_extraction = ExternalTaskSensor(
        task_id='wait_for_data_extraction',
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

    # Define the task dependencies
    wait_for_data_extraction >> run_zenml_pipeline
