from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python_operator import PythonOperator
from airflow.operators.postgres_operator import PostgresOperator
from airflow.hooks.postgres_hook import PostgresHook
from psycopg2.extras import execute_values
import os
import sys

from elasticsearch import Elasticsearch
es = Elasticsearch(['192.168.88.187'])
from datetime import datetime
from elasticsearch_dsl import Document, Date, Integer, Keyword, Text, Float, Boolean,GeoPoint,connections
from tabulate import tabulate
sys.path.append(os.getcwd())



class Query:
    def __init___(self,*args, ** kwargs):
        self.IndexName = args[0]
        self.Size = args[1]
    def get(self,*args,** kwargs):
        documents = es.search(index="{}".format(args[0]),body={"query": {"match_all": {}},"sort": { "_id": "desc"},"size":args[1]})
        df = Select.from_dict(documents).to_pandas()
        return df



def connect():
    connections.create_connection(hosts=['192.168.88.187'])

def get():
    df_es = Query().get('test', 10000)
    df_es = df_es.sort_values('Date_Time', ascending=True)

    last_date_es = [i for i in df_es[-1::]['Date_Time']][0]
    last_index_es = [i for i in df_es[-1::]['_id']][0]
    from airflow.models.taskinstance import task_instance

    task_instance.xcom_push(key='last_date_es', value=last_date_es)
    task_instance.xcom_push(key='last_index_es', value=last_index_es)




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


dag = DAG('get_update_es',
                  default_args=default_args,
                  schedule_interval='@once',
                  start_date=datetime(2017, 3, 20),
                  catchup=False)



task1 = PythonOperator(task_id='connect_es',
                       provide_context=False,
                       python_callable=connect,
                       dag=dag)

task2 = PythonOperator(task_id='get_es',
                       provide_context=False,
                       python_callable=get,
                       dag=dag)

task1 >> task2








# df = pd.read_csv('export_dataframe.csv')
# df= df.sort_values(by=['Date_Time'])
# df.to_sql(name='test2', con=sql_engine, if_exists='append', index = False, chunksize=10000)

