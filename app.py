from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open('./models/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def model_predict_rental_price():
    data = request.form
    rooms = int(data['rooms'])
    sqft = int(data['sqft'])
    
    # Prepare input for prediction
    input_data = np.array([[rooms, sqft]])
    
    # Make prediction
    predicted_rental = model.predict(input_data)[0]
    
    # Return the result as JSON
    return jsonify({
        "rooms": rooms,
        "sqft": sqft,
        "predicted_rental_price": float(predicted_rental)
    })

# For cloud deployment
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)