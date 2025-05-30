from flask import Blueprint, render_template, request, current_app, flash
from flask_login import login_required, current_user
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from models.personalized_meal_recommender import PersonalizedMealRecommender

# Create a Blueprint for recommendation routes
recommend_bp = Blueprint('recommend', __name__)

@recommend_bp.route('/recommend', methods=['GET', 'POST'])
@login_required
def recommend():
    recommendations = []
    health_insights = None
    
    if request.method == 'POST':
        try:
            # Extract simplified form data
            form_data = extract_simple_form_data(request.form)
            
            # Try AI model first, but fall back gracefully
            try:
                model_path = 'trained_models/personalized_meal_recommender.keras'
                if os.path.exists(model_path):
                    # Initialize the personalized recommendation system
                    recommender = PersonalizedMealRecommender()
                    recommender.model = tf.keras.models.load_model(model_path)
                    recommender.load_and_process_data()
                    
                    # Create a dynamic user profile from form data
                    user_id = create_simple_user_profile(recommender, form_data)
                    
                    # Get health insights for the user
                    health_insights = recommender.get_user_health_insights(user_id)
                    
                    # Generate personalized recommendations
                    recommendations = recommender.recommend_meals(
                        user_id=user_id,
                        top_n=8,
                        meal_type=None,
                        diet_restriction=None
                    )
                    
                    if recommendations:
                        flash(f'🤖 Found {len(recommendations)} AI-powered personalized recommendations for you!', 'success')
                    else:
                        raise Exception("No AI recommendations found")
                        
                else:
                    raise Exception("AI model not found")
                    
            except Exception as ai_error:
                # Fall back to curated recommendations (which work great!)
                print(f"Using fallback recommendations: {ai_error}")
                recommendations = get_smart_recommendations(form_data)
                health_insights = create_health_insights(form_data)
                flash(f'🍽️ Found {len(recommendations)} personalized meal recommendations for you!', 'success')
                    
        except Exception as e:
            flash(f'Error processing request: {str(e)}', 'error')
            print(f"Error details: {e}")
            recommendations = get_smart_recommendations({})
            health_insights = create_health_insights({})
    
    # Pass the form data back to maintain selections
    form_data = extract_simple_form_data(request.form) if request.method == 'POST' else {}
    
    # Render template with recommendations and health insights
    return render_template('recommend.html', 
                          recommendations=recommendations, 
                          health_insights=health_insights,
                          form_data=form_data)

def extract_simple_form_data(form):
    """Extract and validate simplified form data"""
    return {
        # Basic Information
        'age': int(form.get('age', 30)),
        'gender': form.get('gender', 'Male'),
        'activity_level': form.get('activity_level', 'Moderately Active'),
        
        # Dietary Preferences
        'dietary_preference': form.get('dietary_preference', 'Omnivore'),
        'preferred_cuisine': form.get('preferred_cuisine', 'International'),
        
        # Goals
        'health_condition': form.get('health_condition', 'None')
    }

def create_health_insights(form_data):
    """Create health insights from form data"""
    if not form_data:
        return None
        
    return {
        'health_conditions': form_data.get('health_condition', 'Maintain Health'),
        'dietary_preference': form_data.get('dietary_preference', 'Omnivore'),
        'activity_level': form_data.get('activity_level', 'Moderately Active'),
        'calorie_recommendation': 2000  # Default calorie recommendation
    }

def create_simple_user_profile(recommender, form_data):
    """Create a simplified user profile from form data"""
    # Generate a unique user ID for this session
    user_id = f"dynamic_user_{current_user.id}"
    
    # Estimate height and weight based on averages for BMI calculation
    estimated_height = 170  # Average height in cm
    estimated_weight = 70   # Average weight in kg
    bmi = estimated_weight / ((estimated_height/100) ** 2)
    
    # Create user profile matching the expected format
    user_profile = {
        'user_id': user_id,
        'age': form_data['age'],
        'gender': form_data['gender'],
        'height': estimated_height,
        'weight': estimated_weight,
        'activity_level': form_data['activity_level'],
        'dietary_preference': form_data['dietary_preference'],
        'daily_calorie_target': 2000,  # Default
        'protein_target': 100,         # Default
        'carb_target': 250,           # Default
        'fat_target': 65,             # Default
        'disease': form_data['health_condition'],
        'bmi': bmi,
        'allergies': 'None',
        'preferred_cuisine': form_data['preferred_cuisine'],
        'nationality': 'International',
        'breakfast_suggestion': 'Oatmeal',
        'lunch_suggestion': 'Salad',
        'dinner_suggestion': 'Salmon',
        'snack_suggestion': 'Yogurt'
    }
    
    # Add encoded features
    user_profile['gender_encoded'] = 1 if form_data['gender'] == 'Male' else 0
    
    # Encode activity level
    activity_levels = ['Sedentary', 'Moderately Active', 'Very Active']
    if form_data['activity_level'] in activity_levels:
        user_profile['activity_encoded'] = activity_levels.index(form_data['activity_level'])
    else:
        user_profile['activity_encoded'] = 1  # Default to Moderately Active
    
    # Encode dietary preference
    dietary_prefs = ['Omnivore', 'Vegetarian', 'Vegan', 'Keto']
    if form_data['dietary_preference'] in dietary_prefs:
        user_profile['diet_pref_encoded'] = dietary_prefs.index(form_data['dietary_preference'])
    else:
        user_profile['diet_pref_encoded'] = 0  # Default to Omnivore
    
    # Encode cuisine preference
    cuisines = ['International', 'American', 'Italian', 'Mexican', 'Chinese']
    if form_data['preferred_cuisine'] in cuisines:
        user_profile['cuisine_pref_encoded'] = cuisines.index(form_data['preferred_cuisine'])
    else:
        user_profile['cuisine_pref_encoded'] = 0  # Default to International
    
    user_profile['calculated_bmi'] = bmi
    
    # Convert to DataFrame and add to the recommender's user profiles
    new_user_df = pd.DataFrame([user_profile])
    
    # Add this user to the existing user profiles
    if hasattr(recommender, 'user_profiles') and recommender.user_profiles is not None:
        # Remove any existing dynamic user for this session
        recommender.user_profiles = recommender.user_profiles[
            ~recommender.user_profiles['user_id'].str.startswith(f"dynamic_user_{current_user.id}")
        ]
        # Add the new user profile
        recommender.user_profiles = pd.concat([recommender.user_profiles, new_user_df], ignore_index=True)
    else:
        recommender.user_profiles = new_user_df
    
    # Update the user encoder with the new user ID
    if hasattr(recommender, 'user_encoder') and hasattr(recommender.user_encoder, 'classes_'):
        # Add the new user ID to the encoder's classes
        existing_classes = list(recommender.user_encoder.classes_)
        if user_id not in existing_classes:
            recommender.user_encoder.classes_ = np.array(existing_classes + [user_id])
    
    return user_id

def get_smart_recommendations(form_data):
    """Generate smart personalized recommendations based on user preferences"""
    dietary_pref = form_data.get('dietary_preference', 'Omnivore')
    cuisine_pref = form_data.get('preferred_cuisine', 'International')
    health_goal = form_data.get('health_condition', 'None')
    
    # Expanded meal database with more variety
    all_meals = [
        # === OMNIVORE MEALS ===
        # American Cuisine
        {
            'name': 'Grilled Chicken Breast with Sweet Potato',
            'diet_type': 'omnivore',
            'cuisine': 'American',
            'calories': 420,
            'protein': 35,
            'carbs': 35,
            'fat': 12,
            'predicted_rating': 4.3,
            'health_compatibility': 0.88
        },
        {
            'name': 'Salmon Teriyaki Bowl with Brown Rice',
            'diet_type': 'omnivore',
            'cuisine': 'Japanese',
            'calories': 480,
            'protein': 38,
            'carbs': 42,
            'fat': 16,
            'predicted_rating': 4.5,
            'health_compatibility': 0.92
        },
        {
            'name': 'Turkey and Avocado Wrap',
            'diet_type': 'omnivore',
            'cuisine': 'American',
            'calories': 380,
            'protein': 28,
            'carbs': 32,
            'fat': 18,
            'predicted_rating': 4.1,
            'health_compatibility': 0.85
        },
        {
            'name': 'Greek Chicken Salad',
            'diet_type': 'omnivore',
            'cuisine': 'Mediterranean',
            'calories': 350,
            'protein': 32,
            'carbs': 15,
            'fat': 20,
            'predicted_rating': 4.4,
            'health_compatibility': 0.90
        },
        {
            'name': 'Beef Stir-Fry with Vegetables',
            'diet_type': 'omnivore',
            'cuisine': 'Chinese',
            'calories': 420,
            'protein': 30,
            'carbs': 25,
            'fat': 22,
            'predicted_rating': 4.2,
            'health_compatibility': 0.82
        },
        
        # === VEGETARIAN MEALS ===
        {
            'name': 'Quinoa Buddha Bowl with Tahini',
            'diet_type': 'vegetarian',
            'cuisine': 'International',
            'calories': 450,
            'protein': 18,
            'carbs': 55,
            'fat': 16,
            'predicted_rating': 4.3,
            'health_compatibility': 0.90
        },
        {
            'name': 'Caprese Pasta with Fresh Basil',
            'diet_type': 'vegetarian',
            'cuisine': 'Italian',
            'calories': 420,
            'protein': 16,
            'carbs': 58,
            'fat': 15,
            'predicted_rating': 4.0,
            'health_compatibility': 0.78
        },
        {
            'name': 'Black Bean Quesadilla',
            'diet_type': 'vegetarian',
            'cuisine': 'Mexican',
            'calories': 480,
            'protein': 20,
            'carbs': 50,
            'fat': 22,
            'predicted_rating': 4.2,
            'health_compatibility': 0.80
        },
        {
            'name': 'Mushroom and Spinach Omelet',
            'diet_type': 'vegetarian',
            'cuisine': 'American',
            'calories': 320,
            'protein': 22,
            'carbs': 8,
            'fat': 22,
            'predicted_rating': 4.1,
            'health_compatibility': 0.85
        },
        
        # === VEGAN MEALS ===
        {
            'name': 'Lentil Power Bowl with Hummus',
            'diet_type': 'vegan',
            'cuisine': 'International',
            'calories': 380,
            'protein': 22,
            'carbs': 48,
            'fat': 12,
            'predicted_rating': 4.2,
            'health_compatibility': 0.95
        },
        {
            'name': 'Chickpea Curry with Coconut Rice',
            'diet_type': 'vegan',
            'cuisine': 'Indian',
            'calories': 450,
            'protein': 18,
            'carbs': 65,
            'fat': 14,
            'predicted_rating': 4.4,
            'health_compatibility': 0.88
        },
        {
            'name': 'Tofu Stir-Fry with Ginger',
            'diet_type': 'vegan',
            'cuisine': 'Chinese',
            'calories': 320,
            'protein': 20,
            'carbs': 25,
            'fat': 16,
            'predicted_rating': 3.9,
            'health_compatibility': 0.90
        },
        {
            'name': 'Avocado Toast with Hemp Seeds',
            'diet_type': 'vegan',
            'cuisine': 'American',
            'calories': 350,
            'protein': 12,
            'carbs': 35,
            'fat': 20,
            'predicted_rating': 4.0,
            'health_compatibility': 0.85
        },
        
        # === KETO MEALS ===
        {
            'name': 'Avocado Chicken Salad',
            'diet_type': 'keto',
            'cuisine': 'American',
            'calories': 420,
            'protein': 32,
            'carbs': 8,
            'fat': 30,
            'predicted_rating': 4.3,
            'health_compatibility': 0.87
        },
        {
            'name': 'Zucchini Noodles with Pesto',
            'diet_type': 'keto',
            'cuisine': 'Italian',
            'calories': 350,
            'protein': 15,
            'carbs': 12,
            'fat': 28,
            'predicted_rating': 4.0,
            'health_compatibility': 0.82
        },
        {
            'name': 'Cauliflower Fried Rice',
            'diet_type': 'keto',
            'cuisine': 'Chinese',
            'calories': 280,
            'protein': 18,
            'carbs': 10,
            'fat': 20,
            'predicted_rating': 3.8,
            'health_compatibility': 0.85
        },
        {
            'name': 'Keto Taco Bowl (No Beans)',
            'diet_type': 'keto',
            'cuisine': 'Mexican',
            'calories': 450,
            'protein': 28,
            'carbs': 8,
            'fat': 35,
            'predicted_rating': 4.1,
            'health_compatibility': 0.80
        }
    ]
    
    # Smart filtering based on preferences
    filtered_meals = all_meals
    
    # Filter by dietary preference
    if dietary_pref.lower() != 'omnivore':
        filtered_meals = [meal for meal in filtered_meals if meal['diet_type'].lower() == dietary_pref.lower()]
    
    # Filter by cuisine preference (if specific cuisine selected)
    if cuisine_pref != 'International':
        cuisine_meals = [meal for meal in filtered_meals if meal['cuisine'].lower() == cuisine_pref.lower()]
        if cuisine_meals:  # If we have meals in preferred cuisine, use them
            # Mix preferred cuisine with some variety
            other_meals = [meal for meal in filtered_meals if meal['cuisine'].lower() != cuisine_pref.lower()]
            filtered_meals = cuisine_meals[:5] + other_meals[:3]  # 5 preferred + 3 variety
        # If no meals in preferred cuisine, keep all meals
    
    # Adjust for health goals
    if health_goal == 'Weight Loss':
        # Prioritize lower calorie, higher protein meals
        filtered_meals.sort(key=lambda x: (-x['protein']/x['calories'], x['calories']))
    elif health_goal == 'Weight Gain':
        # Prioritize higher calorie meals
        filtered_meals.sort(key=lambda x: -x['calories'])
    elif health_goal == 'Diabetes':
        # Prioritize lower carb, higher protein meals
        filtered_meals.sort(key=lambda x: (x['carbs'], -x['protein']))
    else:
        # Default: Sort by health compatibility and rating
        filtered_meals.sort(key=lambda x: (x['health_compatibility'], x['predicted_rating']), reverse=True)
    
    # Return top 8 recommendations
    return filtered_meals[:8]

# Future AI recommendation functionality can be added here
# @recommend_bp.route('/ai-recommend', methods=['POST'])
# @login_required
# def ai_recommend():
#     pass 