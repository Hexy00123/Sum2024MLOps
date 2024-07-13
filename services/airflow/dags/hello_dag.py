# # pipelines/hello_dag.py

# from datetime import datetime

# from airflow import DAG
# from airflow.decorators import task
# from airflow.operators.bash import BashOperator
# from airflow.utils.dates import days_ago
# from datetime import timedelta

# # A DAG represents a workflow, a collection of tasks
# # This DAG is scheduled to print 'hello world' every minute starting from 01.01.2022.
# with DAG(dag_id="hello_world",
#          start_date=days_ago(0),
#          schedule_interval=timedelta(minutes=2)
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

