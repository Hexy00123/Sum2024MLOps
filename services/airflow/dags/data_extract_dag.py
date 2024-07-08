from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
import os

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, '..', '..', '..', 'configs', 'data_version.yaml')
DATA_PY_PATH = os.path.join(BASE_DIR, '..', '..', '..', 'src', 'data.py')
SAMPLE_PATH = os.path.join(BASE_DIR, 'data', 'samples')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'data_extract_dag',
    default_args=default_args,
    description='A simple data extraction DAG',
    schedule_interval=timedelta(minutes=5),
    catchup=False,
)

check_path = BashOperator(
    task_id='check_path',
    bash_command=f'echo {BASE_DIR}',
    dag=dag,
)

extract_data = BashOperator(
    task_id='extract_data',
    bash_command=f'python3 {DATA_PY_PATH}',
    # bash_command='python3 src/data.py',
    dag=dag,
)

validate_data = BashOperator(
    task_id='validate_data',
    bash_command=f'python3 {os.path.join(BASE_DIR, "src", "data_expectations.py")}',
    dag=dag,
)

version_data = BashOperator(
    task_id='version_data',
    bash_command=f'''
    dvc add {SAMPLE_PATH} && \
    TAG="v$(cat {CONFIG_PATH} | grep version | awk '{{print $2}}').0" && \
    git add {SAMPLE_PATH}.dvc && \
    git commit -m "Add data version $TAG" && \
    git push && \
    git tag -a "$TAG" -m "Add data version $TAG" && \
    git push --tags && \
    dvc push
    ''',
    dag=dag,
)

update_version = BashOperator(
    task_id='update_version',
    bash_command=f'''
    python3 -c "import yaml; \
    with open('{CONFIG_PATH}', 'r') as f: config = yaml.safe_load(f); \
    config['version'] += 1; \
    with open('{CONFIG_PATH}', 'w') as f: yaml.safe_dump(config, f)" 
    ''',
    dag=dag,
)

extract_data >> validate_data >> version_data >> update_version