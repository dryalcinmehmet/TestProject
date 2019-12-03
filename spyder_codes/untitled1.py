#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:02:23 2019

@author: doctor
"""


import pandas as pd
import sqlalchemy as sql
import os
import sys
sys.path.append(os.getcwd())
con_st2 = 'mysql://root:-+@127.0.0.1/test'
sql_engine = sql.create_engine(con_st2)

  
df = pd.read_csv('insert.csv')
df= df.sort_values(by=['Date_Time'])
df = df[8:]



df.to_sql(name='test2', con=sql_engine, if_exists='append', index = False, chunksize=10000)


q= "select * from test2"
df_sql = pd.read_sql_query(q, sql_engine)
