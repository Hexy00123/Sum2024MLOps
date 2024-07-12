from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import yaml
import subprocess
import os

from src.data_expectations import validate_initial_data

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
        email_result = subprocess.run(["git", "config", "--global", "user.email"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        name_result = subprocess.run(["git", "config", "--global", "user.name"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        remote_url = subprocess.run(["git", "config", "--get", "remote.origin.url"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        email = email_result.stdout.decode().strip()
        name = name_result.stdout.decode().strip()
        url = remote_url.stdout.decode().strip()
        print(f"Git user email: {email}")
        print(f"Git user name: {name}")
        print(f"Git remote URL: {url}")
    except subprocess.CalledProcessError as e:
        print("Failed to retrieve git config:")
        print(e.stderr.decode())

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
        print(os.chdir('dags'))
        print('dags dir')
        print(os.getcwd())
        print(os.listdir())
        print(e)

def validate_data_sample():
    try:
        validate_initial_data()
    except Exception as e:
        print('Failed!!')
        print(e)

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
        start_date=days_ago(1),
        tags=['example'],
) as dag:
    project_stage = 3

    extract_task = PythonOperator(
        task_id='extract_data_sample',
        python_callable=extract_data_sample,
        op_kwargs={'project_stage': project_stage},
    )

    validate_task = PythonOperator(
        task_id='validate_data_sample',
        python_callable=validate_data_sample,
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
        TAG="v{project_stage}.2"
        
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
        cwd=os.getcwd()
    )

    load_task = BashOperator(
        task_id='load_data_sample',
        bash_command=f'''
            set -e

            TAG="v{project_stage}.0"

            # DVC push
            dvc push

            # Update data version in YAML file
            echo "version: $TAG" > ./configs/data_version.yaml
            ''',
        env={'TAG': f"v{project_stage}.0"},
        cwd=os.getcwd()
    )

    extract_task >> validate_task >> version_task >> load_task
