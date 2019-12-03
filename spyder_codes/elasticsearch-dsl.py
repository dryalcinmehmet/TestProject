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
# Define a default Elasticsearch client
connections.create_connection(hosts=['0.0.0.0'])



class BatteryModel(Document):
    ActivePower = Float()
    Temperature = Float()
    Humidity = Float()
    Weekdays = Integer()
    Vacays   = Boolean()
    Location = GeoPoint()
    Date = Date()
    Place = Text(analyzer='standard', fields={'raw': Keyword()})

    
    class Index:
        name = 'default'
        settings = {
          "number_of_shards": 2,
        }

    def save(self, ** kwargs):
        return super(BatteryModel, self).save(** kwargs)

    def SetValues(self, *args, ** kwargs):
        
    
        self.ActivePower = args[0]
        self.Temperature = args[1]
        self.Humidity    = args[2]
        self.Weekdays    = args[3]
        self.Vacays      = args[4]
        self.Location    = args[5]
        self.Place       = args[6]
        self.Date        = args[7]

    
    def GetValues(self):
        print(self.ActivePower,self.Temperature,self.Humidity,self.Weekdays,self.Vacays,self.Location,self.Date,self.Place)

    
#    def __del__(self):
#        pass
#    
class Query:
    def __init___(self,*args, ** kwargs):
        self.IndexName = args[0]
        self.Size = args[1]
    def get(self,*args,** kwargs):
        documents = es.search(index="{}".format(args[0]),body={"query": {"match_all": {}},"sort": { "_id": "desc"},"size":args[1]})
        df = Select.from_dict(documents).to_pandas()
        return df


class Main:
    def __init__(self):
        print("Starting Process..")

    def __start__(self,*args, ** kwargs):
        
        BatteryObj = BatteryModel(meta={'id': kwargs['_id']})
        BatteryObj.SetValues(*args)
        
        BatteryModel.init(index=kwargs['index'])
        BatteryObj.save(**{'index':'battery','id':1})
        del BatteryObj

df = Query().get('battery',10000)
#q = BatteryModel.get(_id=1)



m = Main()
m.__start__(500,50,40,4,0,"41.055137,28.979530","Taksim",datetime.now(),1,**{'index':'battery','_id':1})

















#
#
#
#print(article.is_published())
#print(connections.get_connection().cluster.health())
