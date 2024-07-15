import os
import subprocess
from datetime import timedelta
import sys

import hydra
from hydra import initialize, compose
import yaml
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from omegaconf import DictConfig

# from data import sample_data, refactor_sample_data

project_root = '/home/pc/Documents/Innopolis/Sum24/MLOps/Project'


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# # @hydra.main(config_path="../configs", config_name="main", version_base=None)
# def extract_data_sample(project_stage):
#     try:
#         subprocess.run(["python3", "src/data.py", f"index={project_stage}"], check=True)
#         print('Data extracted successfully!')
#     except Exception as e:
#         print('Failed!!')
#         print(os.getcwd())
#         print(os.listdir())
#         print(os.chdir('..'))
#         print('up dir')
#         print(os.getcwd())
#         print(os.listdir())
#         # print(os.chdir('dags'))
#         print('dags dir')
#         print(os.getcwd())
#         print(os.listdir())
#         print(e)


# def validate_data_sample():
#     try:
#         subprocess.run(["python3", "src/data_expectations.py"], check=True)
#         print('Data extracted successfully!')
#     except Exception as e:
#         print('Failed!!')
#         print(e)
#         print(type(e))

# def load_data_sample(project_stage):
#     try:
#         TAG = f"v{project_stage}.0"
#         subprocess.run(["dvc", "push"], check=True)
#         with open('./configs/data_version.yaml', 'w') as yaml_file:
#             yaml.dump({"version": TAG}, yaml_file)
#     except Exception as e:
#         print('Failed!!')
#         print(e)


# Define the DAG
with DAG(
        'data_extract_dag',
        default_args=default_args,
        description='A simple data extraction DAG',
        schedule_interval=None,
        start_date=days_ago(0),
        tags=['example'],
) as data_extract_dag:
    with initialize(config_path="../../../configs"):
        cfg = compose(config_name="main")
        project_stage = cfg.index

    print(f"Project stage: {project_stage}")
    extract_task = BashOperator(
        task_id='extract_data_sample',
        bash_command=f'''
        python3 src/data.py index={project_stage}
        ''',
        cwd=project_root,
    )

    validate_task = BashOperator(
        task_id='validate_data_sample',
        bash_command=f'''
        python3 src/data_expectations.py
        ''',
        cwd=project_root,
    )

    version_task = BashOperator(
        task_id='version_data_sample',
        bash_command=f'''
        set -e

        DATA_SAMPLE_PATH="data/samples"
        TAG="v{project_stage}.0"
    
        git config --global --add safe.directory $(pwd)
        
        dvc add $DATA_SAMPLE_PATH

        # Git add and commit
        # Check if $DATA_SAMPLE_PATH.dvc has changed            
        if git diff --cached --quiet --exit-code $DATA_SAMPLE_PATH.dvc; then
            echo "$DATA_SAMPLE_PATH.dvc has not changed. Skipping commit."
            
        else
            # Git add and commit
            git add $DATA_SAMPLE_PATH.dvc
            git commit -m "Add data version $TAG"

            # Check if tag exists and delete it if it does
            if git rev-parse "$TAG" >/dev/null 2>&1; then
                git tag -d "$TAG"
                git push origin --delete "$TAG"
            fi

            # Git tag and push
            git tag -a $TAG -m "Add data version $TAG"
            git push
            git push --tags
        fi
        ''',
        cwd=project_root
    )

    load_task = BashOperator(
        task_id='load_data_sample',
        bash_command=f'''
        set -e
        dvc push
        ''',
        cwd=project_root
    )

    extract_task >> validate_task >> version_task >> load_task
