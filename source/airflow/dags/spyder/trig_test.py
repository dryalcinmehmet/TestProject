#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:26:12 2019

@author: doctor
"""

import pandas as pd
import sqlalchemy as sql
import os
import sys
sys.path.append(os.getcwd())
con_st = 'mysql://root:-+@127.0.0.1/test'
sql_engine = sql.create_engine(con_st)


df = pd.read_csv('osos.csv')
df = df.sort_values('Date_Time', ascending=True)
df=df[['Date_Time','Trafo_id','Active_Energy']]
#df = df.drop_duplicates(subset='Date_Time', keep='first')
#df.to_csv('osos.csv')
df.to_sql(name='test', con=sql_engine, if_exists='append', index = False, chunksize=10000)



q= "select * from test"    
df_sql = pd.read_sql_query(q, sql_engine)


q = "select * from test order by Date_Time desc limit 24"
last_24 =  pd.read_sql_query(q, sql_engine)



qe = "select Date_Time from test order by Date_Time desc limit 1"
last_date_sql = pd.read_sql_query(qe, sql_engine)['Date_Time'][0] 




from source.airflow.dags.spyder.custom_es import Main, Query
from elasticsearch import Elasticsearch
es = Elasticsearch()
from espandas import Espandas



df_sql['indexId'] = (df_sql.index).astype(str)

INDEX = 'test'
TYPE = 'bar_type'
esp = Espandas()
esp.es_write(df_sql, INDEX, TYPE)

df_es=Query().get('test',10000)
df_es = df_es.sort_values('Date_Time', ascending=True)

last_date_es = [i for i in df_es[-1::]['Date_Time']][0]
last_index_es =[i for i in df_es[-1::]['_id']][0]

m = Main()
for j in range(len(last_24['Date_Time'])):
    for i in last_24.loc[j]:
        m.__start__(args[2],50,40,4,0,"41.055137,28.979530",args[1],args[0],1,**{'index':'test','_id':last_index_es+1})