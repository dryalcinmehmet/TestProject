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

  
#df = pd.read_csv('export_dataframe.csv')
#df= df.sort_values(by=['Date_Time'])
#df.to_sql(name='test2', con=sql_engine, if_exists='append', index = False, chunksize=10000)

q= "select * from test2"
df_sql = pd.read_sql_query(q, sql_engine)
last_date_df = [ i for i in pd.read_sql_query(q, sql_engine)[-1:]['Date_Time']][0]

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