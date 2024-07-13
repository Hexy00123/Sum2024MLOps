import os
import subprocess
from datetime import timedelta
import sys

import hydra
import yaml
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from omegaconf import DictConfig


# print(os.getcwd())
# os.chdir("../../../")
# print(os.getcwd())

# sys.path.insert(0, 'home/sshk/project/src')



# from src.data_expectations import validate_initial_data

# Define the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def check_git_config():
    try:
        email_result = subprocess.run(["git", "config", "--global", "user.email"], check=True, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
        name_result = subprocess.run(["git", "config", "--global", "user.name"], check=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        remote_url = subprocess.run(["git", "config", "--get", "remote.origin.url"], check=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        email = email_result.stdout.decode().strip()
        name = name_result.stdout.decode().strip()
        url = remote_url.stdout.decode().strip()
        print(f"Git user email: {email}")
        print(f"Git user name: {name}")
        print(f"Git remote URL: {url}")
    except subprocess.CalledProcessError as e:
        print("Failed to retrieve git config:")
        print(e.stderr.decode())


# @hydra.main(config_path="../configs", config_name="main", version_base=None)
def extract_data_sample(project_stage):
    try:
        subprocess.run(["python3", "src/data.py", f"index={project_stage}"], check=True)
        print('Data extracted successfully!')
    except Exception as e:
        print('Failed!!')
        print(os.getcwd())
        print(os.listdir())
        print(os.chdir('..'))
        print('up dir')
        print(os.getcwd())
        print(os.listdir())
        # print(os.chdir('dags'))
        print('dags dir')
        print(os.getcwd())
        print(os.listdir())
        print(e)


def validate_data_sample():
    try:
        subprocess.run(["python3", "src/data_expectations.py"], check=True)
        print('Data extracted successfully!')
    except Exception as e:
        print('Failed!!')
        print(e)
        print(type(e))

def load_data_sample(project_stage):
    try:
        TAG = f"v{project_stage}.0"
        subprocess.run(["dvc", "push"], check=True)
        with open('./configs/data_version.yaml', 'w') as yaml_file:
            yaml.dump({"version": TAG}, yaml_file)
    except Exception as e:
        print('Failed!!')
        print(e)


# Define the DAG
with DAG(
        'data_extract_dag',
        default_args=default_args,
        description='A simple data extraction DAG',
        schedule_interval=None,
        start_date=days_ago(0),
        tags=['example'],
) as data_extract_dag:
    # project_stage = 3
    # import project stage from config
    # project_stage = hydra.utils.to_absolute_path('configs/main.yaml')
    # with open(project_stage, 'r') as file:
    #     project_stage = yaml.safe_load(file)['index']

    project_stage = 2
    project_root = 'home/sshk/project'

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

        # Print the current working directory
        echo "Current working directory: $(pwd)"
        echo "Files in current directory: $(ls)"

        # Check git configuration
        git config --list

        # Git configuration
        git config --global user.email "sashaki02@gmail.com"
        git config --global user.name "Alexandra Vabnits"
        
        git config --global user.username "sashhhaka"

        # Verify git configuration
        git config --list

        DATA_SAMPLE_PATH="data/samples"
        TAG="v{project_stage}.4"
        
        git config --global --add safe.directory $(pwd)

        # DVC add
        dvc add $DATA_SAMPLE_PATH

        # Git add and commit
        git add $DATA_SAMPLE_PATH.dvc
        git commit -m "Add data version $TAG"

        # Git tag and push
        git tag -a $TAG -m "Add data version $TAG"
        git push
        git push --tags
        ''',
        cwd=project_root
    )

    load_task = BashOperator(
        task_id='load_data_sample',
        bash_command=f'''
            set -e

            TAG="v{project_stage}.0"

            # DVC push
            dvc push

            # Update data version in YAML file
            # echo "version: $TAG" > configs/data_version.yaml
            ''',
        env={'TAG': f"v{project_stage}.0"},
        cwd=project_root
    )


    extract_task >> validate_task >> version_task >> load_task
