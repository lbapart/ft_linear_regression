import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from joblib import dump

data = pd.read_csv('data.csv')
print(data)
plt.scatter(data.km, data.price)
plt.xlabel('km')
plt.ylabel('price')

model = LinearRegression()
x = pd.DataFrame(data['km'])
y = pd.DataFrame(data['price'])
model.fit(x, y)
print(model.predict([[100000]]))

plt.plot(x, model.predict(x), color='red')
print(model.score(x, y))
plt.show()

dump(model, 'model.joblib')