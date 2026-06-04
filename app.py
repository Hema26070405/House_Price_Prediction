from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('house_price_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    sqft_living = float(request.form['sqft_living'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = float(request.form['bathrooms'])
    floors = float(request.form['floors'])

    prediction = model.predict(
        [[sqft_living, bedrooms, bathrooms, floors]]
    )
    price_inr = prediction[0] * 95

    return render_template(
        'index.html',
        prediction_text=f'Predicted House Price: ₹ {price_inr:,.2f}'
          )

if __name__ == '__main__':
    app.run(debug=True)