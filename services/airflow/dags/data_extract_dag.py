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
from omegaconf import DictConfig
import pendulum


# take today's datetime - 5 minutes to avoid issues with scheduling
start_date = pendulum.now(tz="Europe/Moscow").subtract(minutes=5)
start_date = start_date.replace(second=0, microsecond=0)



# from data import sample_data, refactor_sample_data

project_root = '/home/pc/Documents/Innopolis/Sum24/MLOps/Project'


# default_args = {
#     'owner': 'airflow',
#     'depends_on_past': False,
#     'email_on_failure': False,
#     'email_on_retry': False,
#     'retries': 1,
#     'retry_delay': timedelta(minutes=2),
# }

# Define the DAG
with DAG(
        dag_id='data_extract_dag',
        # default_args=default_args,
        description='A simple data extraction DAG',
        schedule_interval=timedelta(minutes=5),
        # schedule_interval=None,
        start_date=start_date,
        tags=['data extraction'],
        # catchup=False,
        is_paused_upon_creation=False,
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
        ''',  # noqa: F541
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
