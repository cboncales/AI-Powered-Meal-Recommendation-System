# AI-Powered Meal Recommendation System

This system uses TensorFlow to build a neural collaborative filtering model that recommends meals to users based on their preferences and nutritional data.

## Overview

The recommendation system:
1. Processes user preferences and meal nutrition data
2. Builds a deep learning model that combines collaborative filtering with content-based features
3. Trains the model on user-meal interactions
4. Generates personalized meal recommendations

## Dataset Information

The system uses multiple datasets:
- `Food_Preference.csv`: Contains user preferences data
- `nutritionverse_dish_metadata3.csv`: Contains dish metadata and nutritional information
- `diet_recommendations_dataset.csv`: Contains diet recommendations

## Model Architecture

- Neural collaborative filtering with embedding layers for users and dishes
- Additional nutritional features for content-based filtering
- Dense neural network to learn complex patterns
- Trained to predict user ratings for dishes

## Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- pandas
- numpy
- scikit-learn
- matplotlib

All dependencies can be installed using:
```
pip install tensorflow pandas scikit-learn matplotlib
```

### Training the Model

To train the model:
```
python train_model.py
```

This will:
- Load and preprocess the datasets
- Build the neural collaborative filtering model
- Train the model on the data
- Save the trained model to the `models` directory
- Generate a training history plot
- Display sample recommendations

### Using the Model for Recommendations

To get meal recommendations for a specific user:
```
python recommend_meals.py --user_id <USER_ID> --top_n 5
```

Parameters:
- `--user_id`: The ID of the user to get recommendations for (required)
- `--top_n`: Number of meal recommendations to generate (default: 5)

## Files and Directory Structure

- `models/meal_recommendation_model.py`: Main model implementation
- `train_model.py`: Script to train the model
- `recommend_meals.py`: Script to generate recommendations
- `datasets/`: Contains the raw datasets
- `models/`: Directory where trained models are saved

## How It Works

1. **Data Processing**: The system loads and preprocesses the datasets, creating synthetic user-dish interactions.
2. **Model Building**: A neural network is built with embeddings for users and dishes, along with nutritional features.
3. **Training**: The model is trained to predict user ratings for dishes.
4. **Recommendation Generation**: For a given user, the system predicts ratings for all available dishes and recommends those with the highest predicted ratings.

## Model Performance

The model's performance can be evaluated through:
- Mean Absolute Error (MAE)
- Training and validation loss curves
- Quality of recommendations

The training history plot is saved to `models/training_history.png` after training. 