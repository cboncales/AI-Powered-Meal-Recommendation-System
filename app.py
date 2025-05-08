from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
import numpy as np
import joblib
import json
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:2001@localhost/MealRecoSys'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a random secret key in production

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# # Load the trained model and encoders
# model = tf.keras.models.load_model("meal_recommendation_model.h5")
# scaler = joblib.load("scaler.pkl")
# label_encoders = joblib.load("label_encoders.pkl")
# target_encoder = joblib.load("target_encoder.pkl")

# Define the User model
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    first_name = db.Column(db.String(50), nullable=True)
    last_name = db.Column(db.String(50), nullable=True)
    
    def set_password(self, password):
        self.password = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password, password)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        first_name = request.form.get('first_name', '')
        last_name = request.form.get('last_name', '')
        
        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('register.html')
            
        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered!', 'error')
            return render_template('register.html')
            
        # Create new user
        new_user = User(email=email, first_name=first_name, last_name=last_name)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Use first name if available, otherwise use email
    display_name = current_user.first_name if current_user.first_name else current_user.email.split('@')[0]
    return render_template('dashboard.html', display_name=display_name)

@app.route('/recommend', methods=['GET', 'POST'])
@login_required
def recommend():
    if request.method == 'POST':
        # Handle form submission (will implement AI recommendations later)
        cuisine = request.form.get('cuisine', '')
        dietary = request.form.get('dietary', '')
        # For now, just return the template with the form
        return render_template('recommend.html')
    
    # For GET requests, just show the recommendation form
    return render_template('recommend.html')

# @app.route('/recommend', methods=['POST'])
# @login_required
# def recommend():
#     try:
#         # Get user input from form
#         dietary_needs = request.form['dietary_needs']
#         past_choices = request.form['past_choices']

#         # Encode categorical inputs
#         dietary_needs_encoded = label_encoders['Dietary_Restrictions'].transform([dietary_needs])[0]

#         # Convert past_choices from JSON string to numeric values
#         past_choices_list = json.loads(past_choices) if past_choices else []
#         past_choices_encoded = np.mean([label_encoders['Diet_Recommendation'].transform([choice])[0] for choice in past_choices_list]) if past_choices_list else 0

#         # Prepare the input for prediction
#         input_data = np.array([[dietary_needs_encoded, past_choices_encoded]])
#         input_data_scaled = scaler.transform(input_data)

#         # Make prediction
#         prediction = model.predict(input_data_scaled)
#         predicted_class = np.argmax(prediction, axis=1) if prediction.shape[1] > 1 else (prediction > 0.5).astype(int)

#         # Decode prediction back to meal name
#         recommended_meal = target_encoder.inverse_transform(predicted_class)[0]

#         return render_template('dashboard.html', recommendation=recommended_meal)

#     except Exception as e:
#         return render_template('dashboard.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
