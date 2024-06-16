import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
#from joblib import dump
from model import MyLinearRegression

data = pd.read_csv('data.csv')
#plt.scatter(data.km, data.price)
#plt.xlabel('km')
#plt.ylabel('price')

#model = LinearRegression()
x = pd.DataFrame(data['km'])
y = pd.DataFrame(data['price'])
#model.fit(x, y)
#print(model.predict([[10000]]))
myModel = MyLinearRegression()
myModel.fit(x, y)
