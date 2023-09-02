# @author Pooja Poonia
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import  SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor  
from sklearn.model_selection import GridSearchCV

data=pd.read_excel("data.xlsx")

data.info()
data.isnull().sum()
category=['Airline','source','Destination']

for i in category:
  print(i,data[i].unique())

data.Date_Of_Journey=data.Date_Of_Journey().str.split('/')

data.Total_Stops.unique()

data.Route=data.Route.str.split('->')

data.Dep_Time=data.Dep_Time.str.split(':')

data.Arrival_Time=data.Arrival_Time.str.split("  ")

data["Arrival Time"]=data.Arrival_Time.str[1]

data.Duration=data.Duration.str.split(' ')

data.Total_Stops=data.Total_Stops.astype('int64')
data.Date=data.Date.astype('int64')
data.Month=data.Month.astype('int64')
data.Year=data.Year.astype('int64')
data.skew()

sc=stadardScaler()

ds_x=data.drop('Price', axis=1)
dataset=sc.fit_trasform(ds_x)

dt=DecisionTreeRegressor()
svr=SVR()
knn=KNeighborsRegressor()
lr=LinearRegressor()

x_train,  x_test, y_train, _test= train_test_split(x,y, test.size=0.3, random_state=42)

for i in  [dt, svr, knn, lr]:

  i.fit(x_train, y_train)
  pred=i.predict(x_test)
  test_score=r2_score(y_test,pred)
  train_score=r2_score(y_train,i.predict(x_train))
  if abs(train_score-test_score)<=0.1:
    print(i)


rfr=RandomForestRegressor()
ad=AdaBoostingRegressor()
gd=GradientBoostingRegressor()

x_train,  x_test, y_train, _test= train_test_split(x,y, test.size=0.3, random_state=42)

for i in  [rfr, ad, gd]:

  i.fit(x_train, y_train)
  pred=i.predict(x_test)
  test_score=r2_score(y_test,pred)
  train_score=r2_score(y_train,i.predict(x_train))
  if abs(train_score-test_score)<=0.2:
    print(i)


model=joblib.load("flight_price.obj")
pred=model.predict(x_test)
predicted_values=pd.DataFrame({'Actual':y_test,'Predicted':pred})

