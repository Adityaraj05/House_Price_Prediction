from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("G:\\house_pricing\\random_forest_model.pkl", 'rb'))


# Load the dataset to get numeric feature names
data = pd.read_csv("G:\\house_pricing\\Housing.csv")


# Select only numeric columns and remove the 'price' column (assuming the last column is the target price)
numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
if 'price' in numeric_features:
    numeric_features.remove('price')  # Remove target variable 'price'

@app.route('/')
def home():
    return render_template('form.html', feature_columns=numeric_features)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data dynamically based on the numeric features
    features = [float(request.form[feature]) for feature in numeric_features]
    
    # Reshape the features array to match the model's expected input
    features = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)
    
    return render_template('form.html', feature_columns=numeric_features, prediction_text='Predicted House Price: ${:.2f}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)


