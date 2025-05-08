from flask import Blueprint, render_template, request
from flask_login import login_required

# Create a Blueprint for recommendation routes
recommend_bp = Blueprint('recommend', __name__)

@recommend_bp.route('/recommend', methods=['GET', 'POST'])
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

# Future AI recommendation functionality can be added here
# @recommend_bp.route('/ai-recommend', methods=['POST'])
# @login_required
# def ai_recommend():
#     pass 