import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("housing.csv")

X = df[['sqft_living', 'bedrooms', 'bathrooms', 'floors']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("house_price_model.pkl", "wb"))

print("Model Saved Successfully")