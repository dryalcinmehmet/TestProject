#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:53:52 2019

@author: doctor
"""

import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="-+",
  database="test"
)

mycursor = mydb.cursor()


mycursor.execute("delete from test")

myresult = mycursor.fetchall()

for x in myresult:
  print(x)
  
  
  
import pandas as pd
import sqlalchemy as sql
import os
import sys
sys.path.append(os.getcwd())
con_st = 'mysql://root:-+@127.0.0.1/test'
sql_engine = sql.create_engine(con_st)

  
df = pd.read_csv('export_dataframe.csv')

import pandas as pd
import numpy as np
from espandas import Espandas



df['indexId'] = (df.index).astype(str)

INDEX = 'foo_index'
TYPE = 'bar_type'
esp = Espandas()
esp.es_write(df, INDEX, TYPE)


documents = es.search(index="foo_index",body={"query": {"match_all": {}},"size":10000})
# Convert the result to Pandas Dataframe
pandas_df = Select.from_dict(documents).to_pandas()




df= df.sort_values(by=['Date_Time'])
df.to_sql(name='test2', con=sql_engine, if_exists='append', index = False, chunksize=10000)

q= "select * from test2"
df_sql = pd.read_sql_query(q, sql_engine)
last_date_df = [ i for i in pd.read_sql_query(q, sql_engine)[-1:]['Date_Time']][0]

from datetime import datetime
from elasticsearch import Elasticsearch
es = Elasticsearch()
from pandasticsearch import Select
import pandas as pd
from elasticsearch import Elasticsearch
from espandas import Espandas




es.indices.create(
    index="test",
    body={
                "mappings": {
                      "properties": {
                          "location": {
                              "type": "geo_point"
                          }
                      }
                }
            }
)








def trigger(last_date_df):
    qe = "select Date_Time from test2 order by Date_Time desc limit 1"
    last_date_sql = pd.read_sql_query(qe, sql_engine)['Date_Time'][0] 
    result=False
    df_sql_last = ''
    if last_date_df < last_date_sql:
        q= '''select * from test2 where Date_Time > "{}" order by Date_Time'''.format(last_date_df)
        df_sql_last = pd.read_sql_query(q, sql_engine)
        print("Select new ok!") 
        result=True
    
    return result, df_sql_last
 

import time
while True:
    try:
        r,df_sql_last=trigger(last_date_df)
        time.sleep(1)
        print(df_sql_last)
    except:
        pass
    
    if  r==True:
        break
    
# df from csv 
# dfsql from sql
# last_date_sql
# last_date_df