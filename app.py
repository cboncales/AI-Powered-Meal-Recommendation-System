from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import joblib
import json

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:2001@localhost/MealRecoSys'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Load the trained model and encoders
model = tf.keras.models.load_model("meal_recommendation_model.h5")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Define the UserPreferences model
class UserPreferences(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, unique=True, nullable=False)
    dietary_needs = db.Column(db.String(100))
    favorite_cuisine = db.Column(db.String(100))
    past_choices = db.Column(db.Text)  # Store as JSON string

# Define the Meal model
class Meal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(100))  # e.g., Vegan, Keto
    ingredients = db.Column(db.Text)  # Store as JSON string

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get user input from form
        dietary_needs = request.form['dietary_needs']
        past_choices = request.form['past_choices']

        # Encode categorical inputs
        dietary_needs_encoded = label_encoders['Dietary_Restrictions'].transform([dietary_needs])[0]

        # Convert past_choices from JSON string to numeric values
        past_choices_list = json.loads(past_choices) if past_choices else []
        past_choices_encoded = np.mean([label_encoders['Diet_Recommendation'].transform([choice])[0] for choice in past_choices_list]) if past_choices_list else 0

        # Prepare the input for prediction
        input_data = np.array([[dietary_needs_encoded, past_choices_encoded]])
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)
        predicted_class = np.argmax(prediction, axis=1) if prediction.shape[1] > 1 else (prediction > 0.5).astype(int)

        # Decode prediction back to meal name
        recommended_meal = target_encoder.inverse_transform(predicted_class)[0]

        return render_template('index.html', recommendation=recommended_meal)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
