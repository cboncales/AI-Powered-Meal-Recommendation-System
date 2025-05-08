import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from models.meal_recommendation_model import MealRecommendationSystem
import argparse

def get_dish_details(dish_id, data_path='datasets'):
    """Get additional details about a dish"""
    dishes_df = pd.read_csv(os.path.join(data_path, 'nutritionverse_dish_metadata3.csv'))
    try:
        dish_info = dishes_df[dishes_df['dish_id'] == dish_id].iloc[0]
        return dish_info
    except:
        return {"dish_id": dish_id}

def main():
    parser = argparse.ArgumentParser(description='Get meal recommendations for a user')
    parser.add_argument('--user_id', type=str, required=True, help='User ID to get recommendations for')
    parser.add_argument('--top_n', type=int, default=5, help='Number of recommendations to return')
    args = parser.parse_args()
    
    user_id = args.user_id
    top_n = args.top_n
    
    # Check if model exists
    model_path = 'models/meal_recommendation_model.keras'
    if not os.path.exists(model_path):
        print("Error: Trained model not found. Please run train_model.py first.")
        return
    
    # Initialize recommendation system
    print(f"Getting meal recommendations for user {user_id}...")
    recommender = MealRecommendationSystem()
    
    # Load data to train the encoders
    recommender.load_data()
    
    # Load the trained model
    recommender.model = load_model(model_path)
    
    try:
        # Get recommendations
        recommendations = recommender.recommend_meals(user_id, top_n=top_n)
        
        if not recommendations:
            print(f"Could not generate recommendations for user {user_id}. User may not exist in training data.")
            return
        
        print(f"\nTop {top_n} meal recommendations for user {user_id}:")
        print("-" * 80)
        
        for i, rec in enumerate(recommendations):
            dish_id = rec['dish_id']
            rating = rec['predicted_rating']
            
            try:
                # Get additional dish details
                dish_info = get_dish_details(dish_id)
                # Get nutrition info (use total_* columns)
                calories = dish_info.get('total_calories', 'N/A')
                protein = dish_info.get('total_protein', 'N/A')
                fats = dish_info.get('total_fats', 'N/A')
                carbs = dish_info.get('total_carbohydrates', 'N/A')
                
                print(f"{i+1}. Dish ID: {dish_id}")
                print(f"   Predicted Rating: {rating:.2f}/5.0")
                print(f"   Nutritional Info: {calories} cal, {protein}g protein, {fats}g fat, {carbs}g carbs")
                print("-" * 80)
            except:
                print(f"{i+1}. Dish ID: {dish_id}")
                print(f"   Predicted Rating: {rating:.2f}/5.0")
                print("-" * 80)
    
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")

if __name__ == "__main__":
    main() 