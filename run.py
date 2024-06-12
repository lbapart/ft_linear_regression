from joblib import load

try:
    model = load('model.joblib')
    print(model.predict([[100000]]))
except FileNotFoundError:
    print('Model is not trained yet')