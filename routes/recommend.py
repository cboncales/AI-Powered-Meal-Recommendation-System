from flask import Blueprint, render_template, request, current_app, flash
from flask_login import login_required, current_user
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from models.meal_recommendation_model import MealRecommendationSystem

# Create a Blueprint for recommendation routes
recommend_bp = Blueprint('recommend', __name__)

# Get available diet types from All_Diets.csv
def get_available_diet_types():
    all_diets_path = 'datasets/All_Diets.csv'
    if os.path.exists(all_diets_path):
        all_diets_df = pd.read_csv(all_diets_path)
        return all_diets_df['Diet_type'].unique().tolist()
    return ['paleo', 'keto', 'vegan', 'vegetarian']  # Defaults if file not found

@recommend_bp.route('/recommend', methods=['GET', 'POST'])
@login_required
def recommend():
    recommendations = []
    
    # Get available diet types for the form
    available_diet_types = get_available_diet_types()
    
    if request.method == 'POST':
        # Get form data
        cuisine = request.form.get('cuisine', '')
        dietary = request.form.get('dietary', '')
        max_calories = request.form.get('max_calories', 2000)
        min_protein = request.form.get('min_protein', 0)
        food_preference = request.form.get('food_preference', 'any')
        beverage = request.form.get('beverage', 'any')
        diet_type = request.form.get('diet_type', None)
        
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
                
                # Get recommendations with specific diet type if selected
                model_recommendations = recommender.recommend_meals(user_id, top_n=6, diet_type=diet_type)
                
                # Get dish details
                dishes_df = pd.read_csv('datasets/nutritionverse_dish_metadata3.csv')
                
                # Load All_Diets data for recipe names and cuisine types
                all_diets_df = pd.read_csv('datasets/All_Diets.csv')
                
                # Map cuisines to more descriptive names
                cuisine_display_map = {
                    'italian': 'Italian',
                    'mexican': 'Mexican',
                    'indian': 'Indian',
                    'chinese': 'Chinese',
                    'american': 'American',
                    'french': 'French',
                    'mediterranean': 'Mediterranean',
                    'japanese': 'Japanese',
                    'thai': 'Thai',
                    'greek': 'Greek',
                    'spanish': 'Spanish',
                    'korean': 'Korean',
                    'vietnamese': 'Vietnamese',
                    'middle eastern': 'Middle Eastern',
                    'filipino': 'Filipino'
                }
                
                # Generate descriptive recipe names based on diet type and cuisine
                def generate_recipe_name(diet_type, cuisine, protein, carbs, dish_id):
                    # Get real recipe names from All_Diets if available
                    matching_recipes = all_diets_df[(all_diets_df['Diet_type'] == diet_type) & 
                                                    (all_diets_df['Cuisine_type'].str.lower() == cuisine.lower())]
                    
                    if not matching_recipes.empty:
                        # Return a real recipe name
                        return matching_recipes.iloc[dish_id % len(matching_recipes)]['Recipe_name']
                    
                    # Fallback to generated names
                    protein_ingredients = {
                        'high': ['Chicken', 'Beef', 'Salmon', 'Tuna', 'Turkey', 'Shrimp', 'Eggs'],
                        'medium': ['Tofu', 'Beans', 'Lentils', 'Quinoa', 'Tempeh'],
                        'low': ['Vegetables', 'Spinach', 'Broccoli', 'Mushrooms']
                    }
                    
                    carb_ingredients = {
                        'low': ['Cauliflower Rice', 'Zucchini Noodles', 'Lettuce Wraps', 'Cabbage'],
                        'medium': ['Brown Rice', 'Quinoa', 'Sweet Potato', 'Whole Grain'],
                        'high': ['Rice', 'Pasta', 'Bread', 'Potatoes', 'Noodles']
                    }
                    
                    cooking_methods = {
                        'keto': ['Grilled', 'Pan-Fried', 'Roasted', 'Baked'],
                        'paleo': ['Grilled', 'Roasted', 'Slow-Cooked', 'Baked'],
                        'vegan': ['Stir-Fried', 'Roasted', 'Steamed', 'Sautéed'],
                        'vegetarian': ['Baked', 'Stir-Fried', 'Sautéed', 'Grilled'],
                        'gluten-free': ['Baked', 'Grilled', 'Poached', 'Steamed'],
                        'mediterranean': ['Grilled', 'Baked', 'Sautéed', 'Roasted']
                    }
                    
                    # Default cooking methods if diet type not in the list
                    default_methods = ['Prepared', 'Cooked', 'Homemade', 'Traditional']
                    
                    # Determine protein level
                    protein_level = 'high' if protein > 30 else 'medium' if protein > 15 else 'low'
                    
                    # Determine carb level
                    carb_level = 'low' if carbs < 15 else 'medium' if carbs < 30 else 'high'
                    
                    # Select appropriate protein ingredient
                    protein_ingredient = np.random.choice(protein_ingredients[protein_level])
                    
                    # Select appropriate carb ingredient
                    carb_ingredient = np.random.choice(carb_ingredients[carb_level])
                    
                    # Select cooking method
                    methods = cooking_methods.get(diet_type.lower(), default_methods)
                    cooking_method = np.random.choice(methods)
                    
                    # Create cuisine-specific dish names
                    cuisine_dishes = {
                        'italian': [f"{cooking_method} {protein_ingredient} with {carb_ingredient}",
                                   f"{protein_ingredient} {carb_ingredient} {cuisine}",
                                   f"{cuisine} {protein_ingredient} with {carb_ingredient}"],
                        'mexican': [f"{protein_ingredient} {cuisine} Bowl with {carb_ingredient}",
                                   f"{cuisine} {protein_ingredient} Tacos",
                                   f"{protein_ingredient} {cuisine} Salad"],
                        'indian': [f"{cuisine} {protein_ingredient} Curry with {carb_ingredient}",
                                  f"{protein_ingredient} {cuisine} Masala",
                                  f"{cuisine} Spiced {protein_ingredient}"],
                        'chinese': [f"{cuisine} {protein_ingredient} Stir-Fry with {carb_ingredient}",
                                   f"{protein_ingredient} {cuisine} Bowl",
                                   f"{cuisine}-Style {protein_ingredient}"],
                        'filipino': [f"{cooking_method} {protein_ingredient} Adobo with {carb_ingredient}",
                                    f"Filipino {protein_ingredient} Sinigang",
                                    f"{protein_ingredient} Kare-Kare with {carb_ingredient}"]
                    }
                    
                    # Get dish names for the cuisine
                    dishes = cuisine_dishes.get(cuisine.lower(), [f"{cooking_method} {protein_ingredient} with {carb_ingredient}",
                                                                f"{cuisine} {protein_ingredient}",
                                                                f"{protein_ingredient} {cuisine} Style"])
                    
                    # Return a dish name
                    return dishes[dish_id % len(dishes)]
                
                # Filter based on cuisine and dietary preferences if specified
                filtered_recommendations = []
                for rec in model_recommendations:
                    dish_id = rec['dish_id']
                    rating = rec['predicted_rating']
                    rec_diet_type = rec['diet_type']
                    
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
                        
                        # Simple mapping for cuisine and diet type (in a real app, these would be in the dataset)
                        cuisines = ['italian', 'mexican', 'indian', 'chinese', 'filipino', 'american', 'french', 'thai']
                        dish_cuisine = cuisines[hash(dish_id) % len(cuisines)]
                        
                        diet_types = ['Regular', 'Vegetarian', 'Vegan', 'Gluten-free']
                        dish_diet = diet_types[hash(dish_id + 1) % len(diet_types)]
                        
                        # Skip if cuisine filter is applied and doesn't match
                        if cuisine and cuisine.lower() != dish_cuisine.lower():
                            continue
                            
                        # Skip if dietary filter is applied and doesn't match
                        if dietary != 'none' and dietary.lower() != dish_diet.lower():
                            continue
                        
                        # Generate a descriptive recipe name
                        recipe_name = generate_recipe_name(rec_diet_type, dish_cuisine, protein, carbs, dish_id)
                        
                        # Add to filtered recommendations
                        filtered_recommendations.append({
                            'dish_id': dish_id,
                            'name': recipe_name,
                            'rating': rating,
                            'calories': calories,
                            'protein': protein,
                            'fats': fats,
                            'carbs': carbs,
                            'cuisine': cuisine_display_map.get(dish_cuisine.lower(), dish_cuisine.capitalize()),
                            'diet_type': dish_diet,
                            'model_diet_type': rec_diet_type,
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
        'beverage': request.form.get('beverage', 'any'),
        'diet_type': request.form.get('diet_type', '')
    }
    
    # Render template with recommendations
    return render_template('recommend.html', 
                          recommendations=recommendations, 
                          form_data=form_data, 
                          available_diet_types=available_diet_types)

# Future AI recommendation functionality can be added here
# @recommend_bp.route('/ai-recommend', methods=['POST'])
# @login_required
# def ai_recommend():
#     pass 