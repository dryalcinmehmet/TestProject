from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python_operator import PythonOperator
from airflow.operators.postgres_operator import PostgresOperator
from airflow.hooks.postgres_hook import PostgresHook
from psycopg2.extras import execute_values
import os,sys
sys.path.append(os.getcwd())
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2018, 4, 15),
    'email': ['example@email.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

dag = DAG('task_csv2sql',
          default_args=default_args,
          schedule_interval='@once',
          start_date=datetime(2017, 3, 20),
          catchup=False)

def csvToPostgres():
    #Open Postgres Connection
    pg_hook = PostgresHook(postgres_conn_id='airflow_db')
    get_postgres_conn = PostgresHook(postgres_conn_id='airflow_db').get_conn()
    curr = get_postgres_conn.cursor("cursor")
    # CSV loading to table.
    with open('/usr/local/airflow/dags/export_dataframe.csv', 'r') as f:
        next(f)
        curr.copy_from(f, 'osos2', sep=',')
        get_postgres_conn.commit()


task1 = PostgresOperator(task_id = 'create_table',
                         sql = ("create table if not exists osos2 " +
                                "(" +
                                    "Trafo_id text," +
                                    "Date_Time timestamp," +
                                    "Active_Energy float"
                                    +
                                ")"),
                         postgres_conn_id='airflow_db',
                         autocommit=True,
                         dag= dag)

task2 = PythonOperator(task_id='csv_to_db',
                   provide_context=False,
                   python_callable=csvToPostgres,
                   dag=dag)


task1 >> task2