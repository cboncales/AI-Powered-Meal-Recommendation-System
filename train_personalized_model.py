import os
import numpy as np
import pandas as pd
import tensorflow as tf
from models.personalized_meal_recommender import PersonalizedMealRecommender

# Check for GPU availability
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

def check_datasets():
    """Check if all required datasets exist"""
    required_files = [
        'All_Diets.csv',
        'daily_food_nutrition_dataset.csv', 
        'detailed_meals_macros_.csv',
        'diet_recommendations_dataset.csv',
        'Food and Calories - Sheet1.csv',
        'Food_and_Nutrition__.csv',
        'Food_Preference.csv',
        'food.csv',
        'nutrition.csv',
        'Food_Recipe.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join('datasets', file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing dataset files: {missing_files}")
        return False
    else:
        print("✅ All dataset files found!")
        return True

def main():
    # Check if datasets exist
    if not check_datasets():
        print("Please ensure all dataset files are in the 'datasets' folder.")
        return
    
    # Initialize the recommendation system
    print("Initializing Personalized Meal Recommendation System...")
    recommender = PersonalizedMealRecommender(data_path='datasets')
    
    try:
        # Load and preprocess data
        print("Loading and processing your actual datasets...")
        train_data, test_data = recommender.load_and_process_data()
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Build the model
        print("Building the personalized health-focused model...")
        recommender.build_model(embedding_dim=64)
        
        # Train the model
        print("Training the model...")
        history = recommender.train(epochs=30, batch_size=32)
        
        # Test recommendations
        print("\n" + "="*80)
        print("TESTING PERSONALIZED MEAL RECOMMENDATIONS")
        print("="*80)
        
        # Get a sample user
        sample_user = recommender.user_profiles['user_id'].iloc[0]
        print(f"\nGenerating recommendations for {sample_user}...")
        
        # Get user health insights
        health_insights = recommender.get_user_health_insights(sample_user)
        if health_insights:
            print(f"\n📊 HEALTH PROFILE for {sample_user}:")
            print(f"   BMI Status: {health_insights['bmi_status']}")
            print(f"   Daily Calories: {health_insights['calorie_recommendation']:.0f}")
            print(f"   Dietary Preference: {health_insights['dietary_preference']}")
            print(f"   Health Conditions: {health_insights['health_conditions']}")
            print(f"   Activity Level: {health_insights['activity_level']}")
        
        # Generate different types of recommendations
        recommendation_types = [
            ("General Recommendations", None, None),
            ("Breakfast Recommendations", "breakfast", None),
            ("Vegetarian Options", None, "vegetarian"),
            ("Keto Diet Options", None, "keto")
        ]
        
        for rec_title, meal_type, diet_restriction in recommendation_types:
            print(f"\n🍽️  {rec_title.upper()}:")
            try:
                recommendations = recommender.recommend_meals(
                    sample_user, 
                    top_n=5, 
                    meal_type=meal_type, 
                    diet_restriction=diet_restriction
                )
                
                if recommendations:
                    for i, rec in enumerate(recommendations, 1):
                        print(f"   {i}. {rec['name']}")
                        print(f"      Rating: {rec['predicted_rating']:.2f}/5 | Diet: {rec['diet_type']}")
                        print(f"      Calories: {rec['calories']:.0f} | Protein: {rec['protein']:.1f}g")
                        print(f"      Health Compatibility: {rec['health_compatibility']:.1%}")
                        print()
                else:
                    print(f"   No recommendations available for these criteria")
                    
            except Exception as e:
                print(f"   Error generating recommendations: {e}")
        
        print("\n" + "="*80)
        print("MODEL TRAINING COMPLETE!")
        print("="*80)
        print("✅ Model saved to: 'trained_models/personalized_meal_recommender.keras'")
        print("✅ Training history plot saved to: 'trained_models/training_history.png'")
        print("\nThe system can now provide personalized meal recommendations based on:")
        print("• User health profiles (BMI, age, activity level)")
        print("• Dietary preferences and restrictions")
        print("• Health conditions and diseases")
        print("• Nutritional goals and targets")
        print("• Cuisine preferences")
        print("\nDataset Summary:")
        print(f"• Users processed: {len(recommender.user_profiles)}")
        print(f"• Recipes processed: {len(recommender.recipe_features)}")
        print(f"• User-recipe interactions: {len(recommender.interactions_df)}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 