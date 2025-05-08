from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required, current_user

# Create a Blueprint for main routes
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return render_template('index.html')

@main_bp.route('/dashboard')
@login_required
def dashboard():
    # Use first name if available, otherwise use email
    display_name = current_user.first_name if current_user.first_name else current_user.email.split('@')[0]
    return render_template('dashboard.html', display_name=display_name) 