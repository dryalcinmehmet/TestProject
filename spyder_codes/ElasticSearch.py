from datetime import datetime
from elasticsearch import Elasticsearch
es = Elasticsearch()
from pandasticsearch import Select
import pandas as pd
from elasticsearch import Elasticsearch
from espandas import Espandas




es.indices.create(
    index="test7",
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


dataset= pd.read_csv('/home/doctor/PycharmProjects/tubitak/battery/spyder_codes/dataset_ready.csv', index_col=0,dtype={'active_power': float,'humidity': float,'temp': float})
dataset.sort_index(inplace=True)
dataset['date']=(dataset.index)
dataset.dtypes
dataset = dataset.reindex(sorted(dataset.columns), axis=1)

dataset['indexId'] =  pd.to_datetime(dataset['date'], format='%Y-%m-%d %H:%M:%S')
dataset.dtypes


dataset.to_sql(name='arge', con=mydb, index=False, chunksize=1000)

#dataset['index']=[i for i in range(0,len(dataset['date']))]
#dataset.set_index('index',inplace=True)

INDEX = 'arge'
TYPE = 'object'
esp = Espandas()
esp.es_write(dataset, INDEX, TYPE)



dataset_24= pd.read_csv('/home/doctor/PycharmProjects/tubitak/battery/spyder_codes/dataset_ready_24.csv', index_col=0,dtype={'active_power': float,'humidity': float,'temp': float})

a={"location":['41.015137,28.979530' ],"value":10,"name":"Beyoglu"}
a=pd.DataFrame.from_dict(a)
a['indexId']=(a.index)

dataset_24.sort_index(inplace=True)
dataset_24['indexId']=(dataset.index)
dataset_24.dtypes
dataset_24 = dataset_24.reindex(sorted(dataset_24.columns), axis=1)
dataset_24['indexId'] =  pd.to_datetime(dataset['indexId'], format='%Y-%m-%d %H:%M:%S')
dataset_24.dtypes



#dataset.set_index('index',inplace=True)
INDEX = 'test7'
TYPE = '_doc'
esp = Espandas()
esp.es_write(a, INDEX, TYPE)



documents = es.search(index="test2",body={"query": {"match_all": {}},"size":10000})
# Convert the result to Pandas Dataframe
pandas_df = Select.from_dict(documents).to_pandas()
pandas_df = pandas_df[['indexId','active_power','temp','humidity']]






d = es.search(index="battery",body={"query": {"match_all": {}},"size":10000})
# Convert the result to Pandas Dataframe
pandas_df = Select.from_dict(d).to_pandas()


type(pandas_df['location'][0])











