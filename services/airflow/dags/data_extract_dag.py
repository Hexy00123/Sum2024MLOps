from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import subprocess
import yaml
import os
# import git

# Define the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# Function to extract a new sample of the data
def extract_data_sample(project_stage):
    # Assuming you have a function in src/data.py for this purpose
    subprocess.run(["python3", "src/data.py",
                   f"index={project_stage}"], check=True)


# Function to validate the sample using Great Expectations
def validate_data_sample():
    # Assuming you have a function in src/data_expectations.py for this purpose
    subprocess.run(["python3", "src/data_expectations.py"], check=True)


# Function to version the sample using DVC
# def version_data_sample(project_stage):
#     DATA_SAMPLE_PATH = "data/samples"
#     TAG = f"v{project_stage}.0"

#     # DVC add
#     subprocess.run(["dvc", "add", DATA_SAMPLE_PATH], check=True)

#     # Git add and commit using GitPython
#     repo = git.Repo('/project')
#     repo.index.add([f"{DATA_SAMPLE_PATH}.dvc"])
#     repo.index.commit(f"Add data version {TAG}")
#     origin = repo.remote(name='origin')
#     origin.push()

#     # Git tag and push tags
#     repo.create_tag(TAG, message=f"Add data version {TAG}")
#     origin.push(tags=True)


# Function to load the sample to the data store>> version_task
def load_data_sample(project_stage):
    TAG = f"v{project_stage}.0"
    # Assuming the remote storage is already configured in your DVC remote
    # No additional code needed if `dvc push` is configured correctly
    # DVC push
    subprocess.run(["dvc", "push"], check=True)

    # Update data version in YAML file
    with open('./configs/data_version.yaml', 'w') as yaml_file:
        yaml.dump({"version": TAG}, yaml_file)


def say_hello():
    print('hello')


def say_bye():
    print('bye!')


# Define the DAG
with DAG(
        'data_extract_dag',
        default_args=default_args,
        description='A simple data extraction DAG',
        schedule_interval=timedelta(minutes=5),
        start_date=days_ago(1),
        tags=['example'],
) as dag:
    project_stage = 2

    extract_task = PythonOperator(
        task_id='extract_data_sample',
        python_callable=extract_data_sample,
        op_kwargs={'project_stage': project_stage},
    )

    validate_task = PythonOperator(
        task_id='validate_data_sample',
        python_callable=validate_data_sample,
    )

    # version_task = PythonOperator(
    #     task_id='version_data_sample',
    #     python_callable=version_data_sample,
    #     op_kwargs={'project_stage': project_stage},
    # )

    load_task = PythonOperator(
        task_id='load_data_sample',
        python_callable=load_data_sample,
        op_kwargs={'project_stage': project_stage},
    )

    simple_task_1 = PythonOperator(
        task_id='say_hello',
        python_callable=say_hello,
    )

    simple_task_2 = PythonOperator(
        task_id='say_bye',
        python_callable=say_bye,
    )

    # Set task dependencies
    # extract_task >> validate_task >> version_task >> load_task
    # extract_task >> validate_task >> load_task
    simple_task_1 >> simple_task_2
