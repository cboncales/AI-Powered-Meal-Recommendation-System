from flask import Flask
import tensorflow as tf
import numpy as np
import joblib
import json
from flask_login import LoginManager

# Import from models
from models.base import init_db
from models.user import User

# Import routes
from routes.auth import auth_bp
from routes.main import main_bp
from routes.recommend import recommend_bp

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:2001@localhost/MealRecoSys'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a random secret key in production

    # Initialize database with app
    db, migrate = init_db(app)

    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(recommend_bp)

    return app

# Create the application instance
app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
