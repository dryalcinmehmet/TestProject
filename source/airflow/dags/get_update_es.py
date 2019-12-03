from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python_operator import PythonOperator
from airflow.operators.postgres_operator import PostgresOperator
from airflow.hooks.postgres_hook import PostgresHook
from psycopg2.extras import execute_values
import pandas as pd
import sqlalchemy as sql
import os
import sys
from datetime import datetime
from elasticsearch import Elasticsearch
es = Elasticsearch()
from pandasticsearch import Select
import pandas as pd
from elasticsearch import Elasticsearch
from espandas import Espandas
from datetime import datetime
from elasticsearch_dsl import Document, Date, Integer, Keyword, Text, Float, Boolean,GeoPoint,connections
from tabulate import tabulate

sys.path.append(os.getcwd())


class BatteryModel(Document):
    ActivePower = Float()
    Temperature = Float()
    Humidity = Float()
    Weekdays = Integer()
    Vacays = Boolean()
    Location = GeoPoint()
    Date = Date()
    Place = Text(analyzer='standard', fields={'raw': Keyword()})

    class Index:
        name = 'default'
        settings = {
            "number_of_shards": 2,
        }

    def save(self, **kwargs):
        return super(BatteryModel, self).save(**kwargs)

    def SetValues(self, *args, **kwargs):
        self.ActivePower = args[0]
        self.Temperature = args[1]
        self.Humidity = args[2]
        self.Weekdays = args[3]
        self.Vacays = args[4]
        self.Location = args[5]
        self.Place = args[6]
        self.Date = args[7]

    def GetValues(self):
        print(self.ActivePower, self.Temperature, self.Humidity, self.Weekdays, self.Vacays, self.Location, self.Date,
              self.Place)

class Query:
    def __init___(self,*args, ** kwargs):
        self.IndexName = args[0]
        self.Size = args[1]
    def get(self,*args,** kwargs):
        documents = es.search(index="{}".format(args[0]),body={"query": {"match_all": {}},"sort": { "_id": "desc"},"size":args[1]})
        df = Select.from_dict(documents).to_pandas()
        return df



class Main:
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

    def __init__(self):
        self.host = "0.0.0.0"
        self.dag = DAG('get_update_es',
                  default_args=self.default_args,
                  schedule_interval='@once',
                  start_date=datetime(2017, 3, 20),
                  catchup=False)
        self.last_date_es = ''



        self.task1 = PythonOperator(task_id='connect_es',
                               provide_context=False,
                               python_callable=self.connect(),
                               dag=self.dag)

        self.task2 = PythonOperator(task_id='get_es',
                               provide_context=False,
                               python_callable=self.get(),
                               dag=self.dag)

        self.task1 >> self.task2



    def connect(self):
        connections.create_connection(hosts=[self.host])

    def get(self,**kwargs):
        last_date_es = Query().get('test',10).sort_values(by=['indexId'])[-1::]['indexId'][0]
        last_index_es = Query().get('test',10).sort_values(by=['indexId'])[-1::]['_id'][0]
        task_instance = kwargs['task_instance']
        task_instance.xcom_push(key='last_date_es', value=last_date_es)
        task_instance.xcom_push(key='last_index_es', value=last_index_es)
        self.last_date_es = task_instance.xcom_pull(task_ids='get_es', key='last_date_es')







# df = pd.read_csv('export_dataframe.csv')
# df= df.sort_values(by=['Date_Time'])
# df.to_sql(name='test2', con=sql_engine, if_exists='append', index = False, chunksize=10000)

