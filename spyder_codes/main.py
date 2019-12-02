import os,sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from datetime import datetime
from pandas import concat
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import random
import sys

sys.path.insert(0, '/home/doctor/Desktop/batarya')
from prepare import PrepareDataset
from scrapping_weather_data import WeatherData

class main():
    
    def __init__(self):
        print("Welcome Machine Learning!")
        self.ProcessBeginning()

        self.dataset_to_LSTM = self.ProcessContinious()
#        
#        
        n_hours = 3
        n_features = 3
        self.LSTM_train_X,self.LSTM_train_y,self.LSTM_test_X,self.LSTM_test_y,self.scaler,self.LSTM_test_24 = PrepareDataset.SupervisorNormalization3D(self.dataset,n_hours,n_features)
#        
        (self.x_train, 
        self.x_test, 
        self.y_train, 
        self.y_test,
        self.x_normal_24,
        self.y_normal_24,
        self.x_24_real,
        self.y_24_real,
        self.dates_float,
        self.sx,
        self.sy) = PrepareDataset.ConvertandNormalization2D(self.dataset)
        
        self.LSTM(n_hours, n_features,self.scaler)
        self.knn()
        self.linear()
        self.SVR()
        self.DecisionTree()
        self.ann()
        self.errors()
        min_rmse = self.ControlRMSE()
        print(min_rmse)
        self.Plot()
   
    
    def ProcessBeginning(self):
         self.dataset=PrepareDataset.ReadDatas() 
         self.dataset.to_json(r'/home/doctor/Desktop/batarya/dataset_ready.json')
         
         
    def ProcessContinious(self):
        #www
        #web scrpping
        temp,humidity = WeatherData.Scrapping()
        
        ###            
        
        self.dataset= pd.read_csv('/home/doctor/Desktop/batarya/dataset_ready_24.csv', index_col=0) ## sql gelecek bu araya  
        self.df_24 = self.dataset.tail(24)

        n = self.df_24.columns[1]
        self.df_24.drop(n, axis = 1, inplace = True)
        self.df_24[n] = temp
        n = self.df_24.columns[2]
        self.df_24.drop(n, axis = 1, inplace = True)
        self.df_24[n] = humidity
        print(self.df_24)
        self.df_24.sort_index(inplace=True)
        
        
        self.js = pd.read_json(r'/home/doctor/Desktop/batarya/dataset_ready.json',orient='Index')
        self.js=self.js.append(self.df_24)
        self.js.to_json(r'/home/doctor/Desktop/batarya/dataset_ready.json')
        self.dataset = self.js
        self.last_datetime = self.dataset[-1:].index[0]
        return self.dataset

    def ControlContainerWithPredictionResults(self):
        pass
    
    
        
              
    def LSTM(self,n_hours, n_features,scaler):
        # design network
        model = Sequential()
        model.add(LSTM(50, input_shape=(self.LSTM_train_X.shape[1], self.LSTM_train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        
        # fit network
        history = model.fit(self.LSTM_train_X, self.LSTM_train_y, epochs=10, batch_size=72, validation_data=(self.LSTM_test_X, self.LSTM_test_y), verbose=2, shuffle=False)
        # make a prediction
        
        yhat = model.predict(self.LSTM_test_X)
        LSTM_test_X = self.LSTM_test_X.reshape((self.LSTM_test_X.shape[0], n_hours*n_features))
        print(LSTM_test_X,LSTM_test_X.shape)
        # invert scaling for forecast
        inv_yhat = concatenate((yhat, LSTM_test_X[:, -2:]), axis=1)
        inv_yhat = self.scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        
        # invert scaling for actual
        LSTM_test_y = self.LSTM_test_y.reshape((len(self.LSTM_test_y), 1))
        print(LSTM_test_y,LSTM_test_y.shape)
        inv_y = concatenate((LSTM_test_y, LSTM_test_X[:, -2:]), axis=1)
        inv_y = self.scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
        
        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)
        
        pyplot.plot(inv_y, label='real')
        pyplot.plot(inv_yhat, label='pred')
        pyplot.legend()
        pyplot.show()
        
        ##for 24
        yhat_24 = model.predict(self.LSTM_test_24)
        LSTM_test_24 = self.LSTM_test_24.reshape((self.LSTM_test_24.shape[0], n_hours*n_features))
        print(LSTM_test_24,LSTM_test_24.shape)
        # invert scaling for forecast
        inv_yhat_24 = concatenate((yhat_24, LSTM_test_24[:, -2:]), axis=1)
        inv_yhat_24 = self.scaler.inverse_transform(inv_yhat_24)
        self.inv_yhat_24 = inv_yhat_24[:,0]
        
        
        
        # invert scaling for actual
        LSTM_test_y_24 = self.LSTM_test_y[-24:].reshape((len(self.LSTM_test_y[-24:]), 1))
        print(LSTM_test_24.shape,LSTM_test_y_24 .shape)
        inv_y_24 = concatenate((LSTM_test_y_24 , LSTM_test_24[:, -2:]), axis=1)
        inv_y_24 = self.scaler.inverse_transform(inv_y_24)
        self.inv_y_24 = inv_y_24[:,0]
        
        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y_24 , inv_yhat_24 ))
        print('Test RMSE: %.3f' % rmse)
        
        pyplot.plot(inv_y_24.reshape(-1,1), label='real')
        pyplot.plot(inv_yhat_24.reshape(-1,1), label='pred')
        pyplot.legend()
        pyplot.show()
        ##
        
        t_h_a=self.dataset.tail(24)  
        t_h_a=t_h_a[['temp','humidity']]
        
        t=[i for i in t_h_a.reset_index(drop=True)['temp']]
        h=[i for i in t_h_a.reset_index(drop=True)['humidity']]
        i=[i for i in self.inv_yhat_24]
    
        
        
        a={'temp':t,'humidity':h,'active_power':i}
        a=pd.DataFrame(a)
            
        a=a[['temp','humidity','active_power','active_power']]
        a.columns = ['temp','humidity','active_power','active_power_y']
      
        print("///////////////////////////",self.inv_yhat_24, self.inv_y_24, self.y_24_real,"//////////////") 
             
        sx=MinMaxScaler()

        sy=MinMaxScaler()
        self.x_normal_24_fromLSTM=sx.fit_transform(a)[:,0:3]
        self.y_normal_24_fromLSTM=sy.fit_transform(a)[:,3]
        
        
        return history
            
    def knn(self):
        from sklearn.neighbors import KNeighborsRegressor
        #find k values
        score_list={}
        for each in range(1,10):
            knn2=KNeighborsRegressor(n_neighbors=each)
            knn2.fit(self.x_train,self.y_train)
            score_list['%d'%each]=knn2.score(self.x_test,self.y_test)
        
        k_max=max(score_list,key=score_list.get) #score list içerisindeki value yu almak için score_list.get bunu kullandık
        neigh = KNeighborsRegressor(n_neighbors=int(k_max))
        neigh.fit(self.x_train,self.y_train) #datayı training etmek
        self.y_pred_knn_24 = neigh.predict(self.x_normal_24_fromLSTM)
        #y_pred_knn_24 = y_pred_knn_24.reshape(-1,1)  #size larını uydurmak için reshape yapılır
        self.knn_score = neigh.score(self.x_test,self.y_test)
        n_d = dict(zip(score_list.keys(), [float(value) for value in score_list.values()])) #zip iki tane listeyi birleştirdi.Dictinary değerleri birbiriyle birleştirdi
        
        k=[] # key'leri listeye çevirdik
        v=[] #value ları listeye çevirdik
        for key,value in n_d.items(): #nd içindeki itemlarda key ve value için 
            k.append(float(key)) #listeye key float ı append komudu ile eklemek
            v.append(float(value))
        
        plt.figure()      
        plt.plot(k,v,color='r')
        plt.xlabel("k values")
        plt.ylabel("Accuracy")
     
        
        self.y_pred_knn_24_real=self.sy.inverse_transform(self.y_pred_knn_24)
        
        plt.figure()
        plt.plot_date(self.dates_float,np.array(self.y_24_real).reshape(-1,1), linestyle='-',xdate=True, ydate=False,label='Real Power')
        plt.plot_date(self.dates_float, self.y_pred_knn_24_real, linestyle='-', xdate=True, ydate=False,label='Prediction')
        plt.gcf().autofmt_xdate()
        plt.ylabel("Kw")
        plt.xlabel("index")
        plt.title("KNN Score: %s"% str(self.knn_score))
        plt.ylabel("Produced Energy/kWh")
        plt.xlabel("Time/Month-Day-Hour ")
        plt.legend(loc='lower right')
        plt.show()
        print("KNN Score: %s"%str(self.knn_score))
        
        
    
    def linear(self):
        
        from sklearn.linear_model import LinearRegression   
        lr=LinearRegression()
        lr.fit(self.x_train, self.y_train)
        self.y_pred_MLR_24 = lr.predict(self.x_normal_24_fromLSTM)
        self.y_pred_MLR_24 = self.y_pred_MLR_24.reshape(-1,1)
        self.linear_score = lr.score(self.x_test,self.y_test)
        
        self.y_pred_MLR_24_real=self.sy.inverse_transform(self.y_pred_MLR_24)
        
        plt.figure()
        plt.plot_date(self.dates_float,np.array(self.y_24_real).reshape(-1,1), linestyle='-',xdate=True, ydate=False,label='Real Power')
        plt.plot_date(self.dates_float, self.y_pred_MLR_24_real, linestyle='-', xdate=True, ydate=False,label='Prediction')
        plt.gcf().autofmt_xdate()
        plt.title("Multi Lineer Regression Score: %s"% str(self.linear_score))
        plt.ylabel("Produced Energy/kWh")
        plt.xlabel("Time/Month-Day-Hour ")
        plt.legend(loc='lower right')
        plt.show()
        print("Multi Lineer Regression Score: %s"%str(self.linear_score))
        
    def SVR(self):
        
        from sklearn.svm import SVR
        
        svm=SVR(C=1.0,epsilon=0.02)
        svm.fit(self.x_train,self.y_train)
        self.y_pred_SVR_24=svm.predict(self.x_normal_24_fromLSTM)
        
        self.SVR_score=svm.score(self.x_test,self.y_test)
        
        
        self.y_pred_SVR_24_real=self.sy.inverse_transform(self.y_pred_SVR_24.reshape(-1,1))
        
        plt.figure()
        plt.plot_date(self.dates_float,np.array(self.y_24_real).reshape(-1,1), linestyle='-',xdate=True, ydate=False,label='Real Power')
        plt.plot_date(self.dates_float, self.y_pred_SVR_24_real, linestyle='-', xdate=True, ydate=False,label='Prediction')
        plt.gcf().autofmt_xdate()
        plt.title("Support Vectorel Machine Regression Score: %s"% str(self.SVR_score))
        plt.ylabel("Produced Energy/kWh")
        plt.xlabel("Time/Month-Day-Hour ")
        plt.legend(loc='lower right')
        plt.show()
        print("Support Vectorel Machine Regression Score: %s"%str(self.SVR_score))
       
    def DecisionTree(self):
        
        from sklearn.tree import DecisionTreeRegressor
        
        
        dt=DecisionTreeRegressor(max_depth=5000)
        dt.fit(self.x_train,self.y_train)
        self.y_pred_tree_24 = dt.predict(self.x_normal_24_fromLSTM)
        self.y_pred_tree_24 = self.y_pred_tree_24.reshape(-1,1)
        
        self.Tree_score=dt.score(self.x_test,self.y_test)
        
        self.y_pred_tree_24_real=self.sy.inverse_transform(self.y_pred_tree_24)
        
        plt.figure()
        plt.plot_date(self.dates_float,np.array(self.y_24_real).reshape(-1,1), linestyle='-',xdate=True, ydate=False, label='Real Power')
        plt.plot_date(self.dates_float, self.y_pred_tree_24_real, linestyle='-', xdate=True, ydate=False,label='Prediction')
        plt.gcf().autofmt_xdate()
        plt.title("Decision Tree Regression Score: %s"% str(self.Tree_score))
        plt.ylabel("Produced Energy/kWh")
        plt.xlabel("Time/Month-Day-Hour ")
        plt.legend(loc='lower right')
        plt.show()
        print("Decision Tree Regression Score: %s"%str(self.Tree_score))

       
    def ann(self):

        from keras.models import Sequential
        from keras.layers import Dense
        import numpy as np
        
        model = Sequential()
        model.add(Dense(4, activation='tanh', input_dim=3))
        model.add(Dense(64, activation='tanh'))
        model.add(Dense(64, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(optimizer='adam', metrics=['accuracy'], loss='mse')
        model.fit(self.x_train, self.y_train, nb_epoch=10,  verbose=2)
        self.y_pred_ANN = model.predict(self.x_test)
       
        self.y_pred_ANN_24=model.predict(self.x_normal_24_fromLSTM)
        self.ann_score=model.evaluate(self.x_test, self.y_test)       
        self.y_pred_ANN_24_real=self.sy.inverse_transform(self.y_pred_ANN_24)
        
        plt.figure()
        plt.plot_date(self.dates_float,np.array(self.y_24_real).reshape(-1,1), linestyle='-',xdate=True, ydate=False,label='Real Power')
        plt.plot_date(self.dates_float, self.y_pred_ANN_24_real, linestyle='-', xdate=True, ydate=False,label='Prediction')
        plt.gcf().autofmt_xdate()
        plt.title("Artificial Neural Network Score: %s"% str(self.ann_score))
        plt.ylabel("Produced Energy/kWh")
        plt.xlabel("Time/Month-Day-Hour ")
        plt.legend(loc='lower right')
        plt.show()
        print("Artificial Neural Network Score: %s"%str(self.ann_score))

    
    def errors(self):
        
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import mean_squared_error
        from math import sqrt

       
        
        self.err_list={"KNN_MAE": mean_absolute_error(self.y_normal_24, self.y_pred_knn_24),
                  "KNN_MSE": mean_squared_error(self.y_normal_24, self.y_pred_knn_24),
                  "KNN_RMSE":sqrt(mean_squared_error(self.y_normal_24,self.y_pred_knn_24)),
                  
                  "MLR_MAE": mean_absolute_error(self.y_normal_24, self.y_pred_MLR_24),
                  "MLR_MSE": mean_squared_error(self.y_normal_24, self.y_pred_MLR_24),
                  "MLR_RMSE": sqrt(mean_squared_error(self.y_normal_24,self.y_pred_MLR_24)),
                  
                  "SVR_MAE": mean_absolute_error(self.y_normal_24, self.y_pred_SVR_24),
                  "SVR_MSE": mean_squared_error(self.y_normal_24, self.y_pred_SVR_24),
                  "SVR_RMSE": sqrt(mean_squared_error(self.y_normal_24,self.y_pred_SVR_24)),
                  
                  
                  "TREE_MAE": mean_absolute_error(self.y_normal_24, self.y_pred_tree_24),
                  "TREE_MSE": mean_squared_error(self.y_normal_24, self.y_pred_tree_24),
                  "TREE_RMSE": sqrt(mean_squared_error(self.y_normal_24,self.y_pred_tree_24)),
                  
                  "ANN_MAE": mean_absolute_error(self.y_normal_24, self.y_pred_ANN_24),
                  "ANN_MSE": mean_squared_error(self.y_normal_24, self.y_pred_ANN_24),
                  "ANN_RMSE": sqrt(mean_squared_error(self.y_normal_24,self.y_pred_ANN_24))
                   }
        
#        y_normal_24=tempdict['y_normal_24']
#        yn24=[i[0] for i in y_normal_24]
#        y_pred_tree_24=tempdict['y_pred_tree_24']
#        ypt24=[round(4,float(i[0])) for i in y_pred_tree_24]
#        
#        for i in y_normal_24:
#            print(round(2,str(i[0])))
#        
#        from sklearn.metrics import accuracy_score
#        knn_acc=accuracy_score(yn24, ypt24)
#        print(knn_acc)
#        
        self.rmse = {
                    'KNN':self.err_list['KNN_RMSE'],
                    'MLR':self.err_list['MLR_RMSE'],
                    'SVR':self.err_list['SVR_RMSE'],
                    'TREE':self.err_list['TREE_RMSE'],
                    'ANN':self.err_list['ANN_RMSE']}
        
        
        
        self.pred_list={"date:":self.dates_float,
                   "y_real":self.y_24_real,
                   "y_pred_knn":self.y_pred_knn_24_real,
                   "y_pred_mlr":self.y_pred_MLR_24_real,
                   "y_pred_svr":self.y_pred_SVR_24_real,
                   "y_pred_tree":self.y_pred_tree_24_real,
                   "y_pred_ann":self.y_pred_ANN_24_real }
        print(self.pred_list)
        
        a=pd.DataFrame(list(zip(self.dates_float, 
                        self.y_24_real[0:24], 
                        self.y_pred_knn_24_real[0:24,0],
                        self.y_pred_MLR_24_real[0:24,0],
                        self.y_pred_SVR_24_real[0:24,0],
                        self.y_pred_tree_24_real[0:24,0],
                        self.y_pred_ANN_24_real[0:24,0])),
              columns=['date','y_real', 'y_pred_knn','y_pred_mlr','y_pred_svr','y_pred_tree','y_pred_ann'])


        a.to_csv("pred_list.csv")  
 
        c=pd.DataFrame(self.err_list, index=[0],) 
        c.to_excel("err.xlsx")

    def ControlRMSE(self):
        print(self.rmse)

        lists = sorted(self.rmse.items()) # sorted by key, return a list of tuples
        
        x, y = zip(*lists) # unpack a list of pairs into two tuples
        
        plt.bar(x, y)
        plt.ylabel("RMSE")
        plt.xlabel("Technices")
        plt.legend(loc='lower right')
        plt.show()
        m=[{'%s'%k:l} for k,l in self.rmse.items() if l == min([i for i in self.rmse.values()])]
        s=[i for i in m[0].keys()][0]
        p = {"self.y_pred_ANN_24_real":self.y_pred_ANN_24_real,
             "self.y_pred_knn_24_real":self.y_pred_knn_24_real,
             "self.y_pred_MLR_24_real":self.y_pred_MLR_24_real,
             "self.y_pred_SVR_24_real":self.y_pred_SVR_24_real,
             "self.y_pred_tree_24_real":self.y_pred_tree_24_real}
        a=[(k,l) for k,l in p.items() if s in k]
        self.todays_predict = a
        pass
        
        
    def Plot(self):
        

        
        plt.figure()
        plt.subplot(711)
        
        plt.plot_date(self.dates_float,np.array(self.y_24_real).reshape(-1,1), linestyle='-',xdate=True, ydate=False,label='Real Power')
        plt.plot_date(self.dates_float, self.inv_yhat_24, linestyle='-', xdate=True, ydate=False,label='LSTM')
        plt.plot_date(self.dates_float, self.y_pred_tree_24_real, linestyle='-', xdate=True, ydate=False,label='DT')
        plt.plot_date(self.dates_float, self.y_pred_ANN_24_real, linestyle='-', xdate=True, ydate=False,label='ANN')
        plt.plot_date(self.dates_float, self.y_pred_MLR_24_real, linestyle='-', xdate=True, ydate=False,label='MLR')
        plt.plot_date(self.dates_float, self.y_pred_knn_24_real, linestyle='-', xdate=True, ydate=False,label='KNN')
        plt.plot_date(self.dates_float, self.y_pred_SVR_24_real, linestyle='-', xdate=True, ydate=False,label='SVM')
        
        plt.gcf().autofmt_xdate()
        plt.ylabel("Produced Energy/kWh")
        plt.xlabel("Time/Month-Day-Hour ")
        plt.legend(loc='lower right')
        
        plt.subplot(712)
        plt.plot_date(self.dates_float,np.array(self.y_24_real).reshape(-1,1), linestyle='-',xdate=True, ydate=False,label='Real Power')
        plt.plot_date(self.dates_float, self.y_pred_ANN_24_real, linestyle='-', xdate=True, ydate=False,label='Prediction')
        plt.gcf().autofmt_xdate()
        plt.title("Artificial Neural Network Score: %s"% str(self.ann_score))
        plt.ylabel("Produced Energy/kWh")
        plt.xlabel("Time/Month-Day-Hour ")
        plt.legend(loc='lower right')
       
        
        plt.subplot(713)
        plt.figure()
        plt.plot_date(self.dates_float,np.array(self.y_24_real).reshape(-1,1), linestyle='-',xdate=True, ydate=False, label='Real Power')
        plt.plot_date(self.dates_float, self.y_pred_tree_24_real, linestyle='-', xdate=True, ydate=False,label='Prediction')
        plt.gcf().autofmt_xdate()
        plt.title("Decision Tree Regression Score: %s"% str(self.Tree_score))
        plt.ylabel("Produced Energy/kWh")
        plt.xlabel("Time/Month-Day-Hour ")
        plt.legend(loc='lower right')
       
        print("Decision Tree Regression Score: %s"%str(self.Tree_score))
        
        plt.subplot(714)
        plt.figure()
        plt.plot_date(self.dates_float,np.array(self.y_24_real).reshape(-1,1), linestyle='-',xdate=True, ydate=False,label='Real Power')
        plt.plot_date(self.dates_float, self.y_pred_SVR_24_real, linestyle='-', xdate=True, ydate=False,label='Prediction')
        plt.gcf().autofmt_xdate()
        plt.title("Support Vectorel Machine Regression Score: %s"% str(self.SVR_score))
        plt.ylabel("Produced Energy/kWh")
        plt.xlabel("Time/Month-Day-Hour ")
        plt.legend(loc='lower right')
       
        plt.subplot(715)
        pyplot.plot(inv_y_24.reshape(-1,1), label='real')
        pyplot.plot(inv_yhat_24.reshape(-1,1), label='pred')
        pyplot.legend()
        
        plt.subplot(716)
        plt.plot_date(self.dates_float,np.array(self.y_24_real).reshape(-1,1), linestyle='-',xdate=True, ydate=False,label='Real Power')
        plt.plot_date(self.dates_float, self.y_pred_knn_24_real, linestyle='-', xdate=True, ydate=False,label='Prediction')
        plt.gcf().autofmt_xdate()
        plt.ylabel("Kw")
        plt.xlabel("index")
        plt.title("KNN Score: %s"% str(self.knn_score))
        plt.ylabel("Produced Energy/kWh")
        plt.xlabel("Time/Month-Day-Hour ")
        plt.legend(loc='lower right')
        
        plt.subplot(717)
        plt.plot_date(self.dates_float,np.array(self.y_24_real).reshape(-1,1), linestyle='-',xdate=True, ydate=False,label='Real Power')
        plt.plot_date(self.dates_float, self.y_pred_MLR_24_real, linestyle='-', xdate=True, ydate=False,label='Prediction')
        plt.gcf().autofmt_xdate()
        plt.ylabel("Kw")
        plt.xlabel("index")
        plt.title("MLR Score: %s"% str(self.linear_score))
        plt.ylabel("Produced Energy/kWh")
        plt.xlabel("Time/Month-Day-Hour ") 
        
if __name__ =="__main__":
    i=main()
    tempdict=i.__dict__



















