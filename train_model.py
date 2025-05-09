import os
import tensorflow as tf
from models.meal_recommendation_model import MealRecommendationSystem

# Check for GPU availability
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Initialize the recommendation system
print("Initializing Meal Recommendation System...")
recommender = MealRecommendationSystem(data_path='datasets')

# Load and preprocess data
print("Loading and preprocessing data (including All_Diets.csv)...")
train_df, test_df = recommender.load_data()

# Build the model
print("Building the enhanced model with diet type features...")
recommender.build_model(embedding_dim=32)

# Train the model
print("Training the model...")
history = recommender.train(epochs=15, batch_size=64)

# Generate recommendations for a sample user
sample_user = train_df['user_id'].iloc[0]
print(f"\nGenerating meal recommendations for user {sample_user}...")

# Try different diet types for the same user
diet_types = ['paleo', 'keto', 'vegan', 'vegetarian']

for diet_type in diet_types:
    print(f"\n{diet_type.upper()} DIET RECOMMENDATIONS:")
    recommendations = recommender.recommend_meals(sample_user, top_n=3, diet_type=diet_type)
    
    for i, rec in enumerate(recommendations):
        print(f"{i+1}. Dish ID: {rec['dish_id']}, Predicted Rating: {rec['predicted_rating']:.2f}, Diet Type: {rec['diet_type']}")

print("\nModel training complete!")
print("The trained model has been saved to 'models/meal_recommendation_model.keras'")
print("Training history plot has been saved to 'models/training_history.png'") 