from flask import Blueprint, render_template, request, current_app, flash
from flask_login import login_required, current_user
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from models.meal_recommendation_model import MealRecommendationSystem

# Create a Blueprint for recommendation routes
recommend_bp = Blueprint('recommend', __name__)

@recommend_bp.route('/recommend', methods=['GET', 'POST'])
@login_required
def recommend():
    recommendations = []
    
    if request.method == 'POST':
        # Get form data
        cuisine = request.form.get('cuisine', '')
        dietary = request.form.get('dietary', '')
        max_calories = request.form.get('max_calories', 2000)
        min_protein = request.form.get('min_protein', 0)
        food_preference = request.form.get('food_preference', 'any')
        beverage = request.form.get('beverage', 'any')
        
        try:
            # Convert string values to appropriate types
            try:
                max_calories = float(max_calories)
                min_protein = float(min_protein)
            except ValueError:
                max_calories = 2000
                min_protein = 0
            
            # Initialize the recommendation system
            recommender = MealRecommendationSystem()
            
            # Load data to train the encoders
            recommender.load_data()
            
            # Load the trained model
            model_path = 'models/meal_recommendation_model.keras'
            if os.path.exists(model_path):
                recommender.model = tf.keras.models.load_model(model_path)
                
                # Map user to a participant based on food and beverage preferences
                # This makes the mapping more personalized based on actual preferences
                participant_df = pd.read_csv('datasets/Food_Preference.csv')
                
                # Filter based on food preference if specified
                if food_preference != 'any':
                    food_type = 'Traditional food' if food_preference == 'traditional' else 'Western Food'
                    participant_df = participant_df[participant_df['Food'] == food_type]
                
                # Filter based on beverage preference if specified
                if beverage != 'any':
                    beverage_type = 'Fresh Juice' if beverage == 'fresh' else 'Carbonated drinks'
                    participant_df = participant_df[participant_df['Juice'] == beverage_type]
                
                # If no matches, use all participants
                if len(participant_df) == 0:
                    participant_df = pd.read_csv('datasets/Food_Preference.csv')
                
                # Get list of matching participant IDs
                participant_ids = participant_df['Participant_ID'].unique()
                
                # Pick a participant ID based on the user's id
                user_index = hash(current_user.id) % len(participant_ids)
                user_id = participant_ids[user_index]
                
                # Get recommendations
                model_recommendations = recommender.recommend_meals(user_id, top_n=6)
                
                # Get dish details
                dishes_df = pd.read_csv('datasets/nutritionverse_dish_metadata3.csv')
                
                # Filter based on cuisine and dietary preferences if specified
                filtered_recommendations = []
                for rec in model_recommendations:
                    dish_id = rec['dish_id']
                    rating = rec['predicted_rating']
                    
                    try:
                        # Get dish details
                        dish_info = dishes_df[dishes_df['dish_id'] == dish_id].iloc[0]
                        
                        # Get nutrition info
                        calories = dish_info.get('total_calories', 0)
                        protein = dish_info.get('total_protein', 0)
                        fats = dish_info.get('total_fats', 0)
                        carbs = dish_info.get('total_carbohydrates', 0)
                        
                        # Skip if calories are higher than max_calories
                        if calories > max_calories:
                            continue
                            
                        # Skip if protein is lower than min_protein
                        if protein < min_protein:
                            continue
                        
                        # Add to recommendations list
                        dish_name = dish_info.get('name', f'Dish {dish_id}')
                        
                        # Simple mapping for cuisine and diet type (in a real app, these would be in the dataset)
                        cuisines = ['Italian', 'Mexican', 'Indian', 'Chinese']
                        dish_cuisine = cuisines[hash(dish_id) % len(cuisines)]
                        
                        diet_types = ['Regular', 'Vegetarian', 'Vegan', 'Gluten-free']
                        dish_diet = diet_types[hash(dish_id + 1) % len(diet_types)]
                        
                        # Skip if cuisine filter is applied and doesn't match
                        if cuisine and cuisine.lower() != dish_cuisine.lower():
                            continue
                            
                        # Skip if dietary filter is applied and doesn't match
                        if dietary != 'none' and dietary.lower() != dish_diet.lower():
                            continue
                        
                        # Add to filtered recommendations
                        filtered_recommendations.append({
                            'dish_id': dish_id,
                            'name': dish_name,
                            'rating': rating,
                            'calories': calories,
                            'protein': protein,
                            'fats': fats,
                            'carbs': carbs,
                            'cuisine': dish_cuisine,
                            'diet_type': dish_diet,
                            'cook_time': f"{int(20 + dish_id % 40)} mins"
                        })
                    except Exception as e:
                        continue
                
                recommendations = filtered_recommendations[:3]  # Limit to top 3
                
                if not recommendations:
                    flash('No meals matching your criteria were found. Try adjusting your preferences.', 'info')
            else:
                flash('Recommendation model not found. Please contact the administrator.', 'error')
        except Exception as e:
            flash(f'Error generating recommendations: {str(e)}', 'error')
    
    # Pass the form data back to the template to maintain selections
    form_data = {
        'cuisine': request.form.get('cuisine', ''),
        'dietary': request.form.get('dietary', 'none'),
        'max_calories': request.form.get('max_calories', '800'),
        'min_protein': request.form.get('min_protein', '30'),
        'food_preference': request.form.get('food_preference', 'any'),
        'beverage': request.form.get('beverage', 'any')
    }
    
    # Render template with recommendations
    return render_template('recommend.html', recommendations=recommendations, form_data=form_data)

# Future AI recommendation functionality can be added here
# @recommend_bp.route('/ai-recommend', methods=['POST'])
# @login_required
# def ai_recommend():
#     pass 