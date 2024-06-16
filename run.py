import pickle

gradient = 0
slope = 0

def predict(independent):
    result = gradient + (slope * independent)
    return result

try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
        gradient = model.gradient
        slope = model.slope
except:
    print("Model not found. Using default values.")

number = input("Enter a number: ")
if number.isnumeric():
    print(predict(int(number)))