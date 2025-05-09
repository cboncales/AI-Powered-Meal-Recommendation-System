import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MealRecommendationSystem:
    def __init__(self, data_path='datasets'):
        self.data_path = data_path
        self.model = None
        self.user_encoder = LabelEncoder()
        self.dish_encoder = LabelEncoder()
        self.diet_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and preprocess the datasets"""
        print("Loading datasets...")
        
        # Load Food Preference data (user preferences)
        preferences_df = pd.read_csv(os.path.join(self.data_path, 'Food_Preference.csv'))
        
        # Load dish metadata
        dishes_df = pd.read_csv(os.path.join(self.data_path, 'nutritionverse_dish_metadata3.csv'))
        
        # Load diet recommendations
        diets_df = pd.read_csv(os.path.join(self.data_path, 'diet_recommendations_dataset.csv'))
        
        # Load All_Diets dataset
        all_diets_df = pd.read_csv(os.path.join(self.data_path, 'All_Diets.csv'))
        
        # Basic data cleaning
        preferences_df = preferences_df.dropna()
        dishes_df = dishes_df.dropna()
        diets_df = diets_df.dropna()
        all_diets_df = all_diets_df.dropna()
        
        # Process All_Diets data to integrate with our model
        # We'll create a mapping from diet types to dishes
        diet_types = all_diets_df['Diet_type'].unique()
        self.diet_type_encoder = LabelEncoder()
        diet_type_encoded = self.diet_type_encoder.fit_transform(all_diets_df['Diet_type'])
        all_diets_df['diet_type_encoded'] = diet_type_encoded
        
        # Extract cuisine types for later use
        cuisine_types = all_diets_df['Cuisine_type'].unique()
        self.cuisine_encoder = LabelEncoder()
        cuisine_encoded = self.cuisine_encoder.fit_transform(all_diets_df['Cuisine_type'])
        all_diets_df['cuisine_encoded'] = cuisine_encoded
        
        # Create user-dish interaction matrix
        print("Creating user-dish interactions...")
        
        # Get unique users and dishes
        unique_users = preferences_df['Participant_ID'].unique()
        unique_dishes = dishes_df['dish_id'].unique()
        
        # Create a sample interaction dataset
        # In a real system, this would be actual user ratings/interactions with dishes
        # For this demo, we'll create synthetic interactions with diet preferences integrated
        num_interactions = min(15000, len(unique_users) * 20)  # Increased interactions
        
        user_ids = np.random.choice(unique_users, num_interactions)
        dish_ids = np.random.choice(unique_dishes, num_interactions)
        
        # Generate ratings (1-5) with some preference patterns
        ratings = np.random.randint(1, 6, num_interactions)
        
        # Create interactions dataframe
        interactions_df = pd.DataFrame({
            'user_id': user_ids,
            'dish_id': dish_ids,
            'rating': ratings
        })
        
        # Enhance interactions with diet information based on user preferences
        # Map users to preferred diets using food and dessert preferences
        diet_mapping = {
            'Traditional food': ['paleo', 'mediterranean', 'keto'],
            'Western Food': ['vegan', 'vegetarian', 'gluten-free']
        }
        
        # Add diet type to user profiles using preferences
        user_diet_prefs = {}
        for _, row in preferences_df.iterrows():
            user_id = row['Participant_ID']
            food_pref = row['Food']
            dessert_pref = row['Dessert']
            
            # Assign diet types based on preferences
            if food_pref in diet_mapping:
                # Select a diet type based on food preference
                possible_diets = diet_mapping[food_pref]
                # Use dessert preference to further narrow down diet
                if dessert_pref == 'Yes':
                    diet_idx = 0  # More permissive diet
                elif dessert_pref == 'No':
                    diet_idx = len(possible_diets) - 1  # More restrictive diet
                else:  # 'Maybe'
                    diet_idx = np.random.randint(0, len(possible_diets))
                
                user_diet_prefs[user_id] = possible_diets[diet_idx]
            else:
                # Default to random diet type
                user_diet_prefs[user_id] = np.random.choice(diet_types)
        
        # Add diet type to interactions
        diet_types_list = []
        cuisine_types_list = []
        
        for _, row in interactions_df.iterrows():
            user_id = row['user_id']
            
            # Get user's diet preference
            user_diet = user_diet_prefs.get(user_id, np.random.choice(diet_types))
            diet_types_list.append(user_diet)
            
            # Assign a cuisine type based on the diet
            diet_cuisines = all_diets_df[all_diets_df['Diet_type'] == user_diet]['Cuisine_type'].unique()
            if len(diet_cuisines) > 0:
                cuisine = np.random.choice(diet_cuisines)
            else:
                cuisine = np.random.choice(cuisine_types)
            cuisine_types_list.append(cuisine)
        
        interactions_df['diet_type'] = diet_types_list
        interactions_df['cuisine_type'] = cuisine_types_list
        
        # Encode categorical features
        interactions_df['user_id_encoded'] = self.user_encoder.fit_transform(interactions_df['user_id'])
        interactions_df['dish_id_encoded'] = self.dish_encoder.fit_transform(interactions_df['dish_id'])
        interactions_df['diet_type_encoded'] = self.diet_type_encoder.fit_transform(interactions_df['diet_type'])
        
        # Get dish features to enrich the model
        dish_features = dishes_df.copy()
        # Select relevant nutritional features 
        nutrition_cols = ['total_calories', 'total_protein', 'total_fats', 'total_carbohydrates']
        # Rename columns to simpler names for easier access
        dish_features = dish_features[['dish_id'] + nutrition_cols].dropna()
        dish_features = dish_features.rename(columns={
            'total_calories': 'Calories',
            'total_protein': 'Protein',
            'total_fats': 'Fats',
            'total_carbohydrates': 'Carbs'
        })
        
        # Normalize nutritional features
        dish_features[['Calories', 'Protein', 'Fats', 'Carbs']] = self.scaler.fit_transform(dish_features[['Calories', 'Protein', 'Fats', 'Carbs']])
        
        # Merge dish features with interactions
        merged_df = pd.merge(
            interactions_df, 
            dish_features,
            left_on='dish_id',
            right_on='dish_id',
            how='inner'
        )
        
        # Add nutritional preferences from All_Diets to enhance our model
        nutrition_preferences = {}
        for diet_type in diet_types:
            # Get average nutritional values for each diet type
            diet_data = all_diets_df[all_diets_df['Diet_type'] == diet_type]
            avg_protein = diet_data['Protein(g)'].mean()
            avg_carbs = diet_data['Carbs(g)'].mean()
            avg_fat = diet_data['Fat(g)'].mean()
            
            nutrition_preferences[diet_type] = {
                'protein': avg_protein,
                'carbs': avg_carbs,
                'fat': avg_fat
            }
        
        # Store for later use
        self.nutrition_preferences = nutrition_preferences
        
        # Split data
        self.train_df, self.test_df = train_test_split(merged_df, test_size=0.2, random_state=42)
        
        print(f"Training data shape: {self.train_df.shape}")
        print(f"Testing data shape: {self.test_df.shape}")
        
        return self.train_df, self.test_df
    
    def build_model(self, embedding_dim=50):
        """Build a neural collaborative filtering model with additional features"""
        # Number of unique users and dishes
        n_users = len(self.user_encoder.classes_)
        n_dishes = len(self.dish_encoder.classes_)
        n_diets = len(self.diet_type_encoder.classes_)
        
        # User input and embedding
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(n_users, embedding_dim, name='user_embedding')(user_input)
        user_vec = Flatten(name='user_flatten')(user_embedding)
        
        # Dish input and embedding
        dish_input = Input(shape=(1,), name='dish_input')
        dish_embedding = Embedding(n_dishes, embedding_dim, name='dish_embedding')(dish_input)
        dish_vec = Flatten(name='dish_flatten')(dish_embedding)
        
        # Diet type input and embedding
        diet_input = Input(shape=(1,), name='diet_input')
        diet_embedding = Embedding(n_diets, embedding_dim // 2, name='diet_embedding')(diet_input)
        diet_vec = Flatten(name='diet_flatten')(diet_embedding)
        
        # Nutritional features input
        nutrition_input = Input(shape=(4,), name='nutrition_input')  # 4 nutritional features
        
        # Combine embeddings and features
        concat = Concatenate()([user_vec, dish_vec, diet_vec, nutrition_input])
        
        # Dense layers
        dense1 = Dense(128, activation='relu')(concat)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        dense3 = Dense(32, activation='relu')(dropout2)
        
        # Output layer
        output = Dense(1)(dense3)
        
        # Create model
        model = Model(
            inputs=[user_input, dish_input, diet_input, nutrition_input],
            outputs=output
        )
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        self.model = model
        print(model.summary())
        return model
    
    def train(self, epochs=20, batch_size=64):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Prepare inputs
        user_train = self.train_df['user_id_encoded'].values
        dish_train = self.train_df['dish_id_encoded'].values
        diet_train = self.train_df['diet_type_encoded'].values
        nutrition_train = self.train_df[['Calories', 'Protein', 'Fats', 'Carbs']].values
        
        user_test = self.test_df['user_id_encoded'].values
        dish_test = self.test_df['dish_id_encoded'].values
        diet_test = self.test_df['diet_type_encoded'].values
        nutrition_test = self.test_df[['Calories', 'Protein', 'Fats', 'Carbs']].values
        
        # Target values
        y_train = self.train_df['rating'].values
        y_test = self.test_df['rating'].values
        
        # Train the model
        history = self.model.fit(
            [user_train, dish_train, diet_train, nutrition_train],
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([user_test, dish_test, diet_test, nutrition_test], y_test),
            verbose=1
        )
        
        # Save the model with proper extension
        self.model.save('models/meal_recommendation_model.keras')
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        plt.show()
        
        return history
    
    def recommend_meals(self, user_id, top_n=5, diet_type=None):
        """Generate meal recommendations for a user"""
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        # Get user index
        try:
            user_idx = self.user_encoder.transform([user_id])[0]
        except:
            print(f"User {user_id} not found in training data.")
            return []
        
        # Get all dishes
        dish_indices = np.arange(len(self.dish_encoder.classes_))
        dishes = self.dish_encoder.inverse_transform(dish_indices)
        
        # Get diet type
        if diet_type is None:
            # Try to find a diet type from user's training data
            if user_id in self.train_df['user_id'].values:
                user_diets = self.train_df[self.train_df['user_id'] == user_id]['diet_type'].value_counts()
                if not user_diets.empty:
                    diet_type = user_diets.index[0]
                else:
                    diet_type = 'paleo'  # Default to paleo if no preference found
            else:
                diet_type = 'paleo'  # Default to paleo if user not in training data
        
        try:
            diet_idx = self.diet_type_encoder.transform([diet_type])[0]
        except:
            # If diet type not found, default to first diet type
            diet_type = self.diet_type_encoder.classes_[0]
            diet_idx = 0
            
        # Load dish features
        dishes_df = pd.read_csv(os.path.join(self.data_path, 'nutritionverse_dish_metadata3.csv'))
        dish_features = dishes_df.copy()
        nutrition_cols = ['total_calories', 'total_protein', 'total_fats', 'total_carbohydrates']
        dish_features = dish_features[['dish_id'] + nutrition_cols].dropna()
        dish_features = dish_features.rename(columns={
            'total_calories': 'Calories',
            'total_protein': 'Protein',
            'total_fats': 'Fats',
            'total_carbohydrates': 'Carbs'
        })
        
        # Create prediction input
        user_input = np.array([user_idx] * len(dishes))
        dish_input = dish_indices
        diet_input = np.array([diet_idx] * len(dishes))
        
        # Get nutritional data for all dishes
        nutrition_input = []
        for dish in dishes:
            dish_data = dish_features[dish_features['dish_id'] == dish][['Calories', 'Protein', 'Fats', 'Carbs']].values
            if len(dish_data) > 0:
                nutrition_input.append(dish_data[0])
            else:
                # Use average values if dish not found
                nutrition_input.append(dish_features[['Calories', 'Protein', 'Fats', 'Carbs']].mean().values)
        
        nutrition_input = np.array(nutrition_input)
        # Normalize nutritional features
        nutrition_input = self.scaler.transform(nutrition_input)
        
        # Predict ratings
        predictions = self.model.predict([user_input, dish_input, diet_input, nutrition_input], verbose=0).flatten()
        
        # Get top N recommendations
        dish_indices = np.argsort(-predictions)[:top_n]
        recommended_dishes = dishes[dish_indices]
        predicted_ratings = predictions[dish_indices]
        
        # Add the diet type to recommendations
        recommendations = [
            {"dish_id": dish, "predicted_rating": rating, "diet_type": diet_type}
            for dish, rating in zip(recommended_dishes, predicted_ratings)
        ]
        
        return recommendations

if __name__ == "__main__":
    # Initialize the recommendation system
    recommender = MealRecommendationSystem()
    
    # Load and preprocess data
    train_df, test_df = recommender.load_data()
    
    # Build the model
    recommender.build_model(embedding_dim=32)
    
    # Train the model (reduce epochs for testing)
    history = recommender.train(epochs=10, batch_size=64)
    
    # Generate recommendations for a sample user
    sample_user = train_df['user_id'].iloc[0]
    recommendations = recommender.recommend_meals(sample_user, top_n=5)
    
    print(f"\nTop 5 meal recommendations for user {sample_user}:")
    for i, rec in enumerate(recommendations):
        print(f"{i+1}. Dish ID: {rec['dish_id']}, Predicted Rating: {rec['predicted_rating']:.2f}, Diet Type: {rec['diet_type']}") 