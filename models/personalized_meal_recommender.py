import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout, BatchNormalization
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PersonalizedMealRecommender:
    def __init__(self, data_path='datasets'):
        self.data_path = data_path
        self.model = None
        self.user_encoder = LabelEncoder()
        self.recipe_encoder = LabelEncoder()
        self.cuisine_encoder = LabelEncoder()
        self.diet_encoder = LabelEncoder()
        self.activity_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.nutrition_scaler = MinMaxScaler()
        
        # Store encoders and data for recommendations
        self.users_df = None
        self.recipes_df = None
        self.user_profiles = None
        self.recipe_features = None
        
    def load_and_process_data(self):
        """Load and process all 10 actual datasets"""
        print("Loading actual datasets...")
        
        # Dataset 1: All_Diets.csv - Diet/Recipe nutritional data
        diet_recipes_df = pd.read_csv(os.path.join(self.data_path, 'All_Diets.csv'))
        
        # Dataset 2: daily_food_nutrition_dataset.csv - User food consumption history
        user_consumption_df = pd.read_csv(os.path.join(self.data_path, 'daily_food_nutrition_dataset.csv'))
        
        # Dataset 3: detailed_meals_macros_.csv - User profiles with health data
        user_profiles_df = pd.read_csv(os.path.join(self.data_path, 'detailed_meals_macros_.csv'))
        
        # Dataset 4: diet_recommendations_dataset.csv - Patient medical data
        patient_data_df = pd.read_csv(os.path.join(self.data_path, 'diet_recommendations_dataset.csv'))
        
        # Dataset 5: Food and Calories - Sheet1.csv - Basic food calories
        food_calories_df = pd.read_csv(os.path.join(self.data_path, 'Food and Calories - Sheet1.csv'))
        
        # Dataset 6: Food_and_Nutrition__.csv - Diet recommendations
        diet_recommendations_df = pd.read_csv(os.path.join(self.data_path, 'Food_and_Nutrition__.csv'))
        
        # Dataset 7: Food_Preference.csv - Food preferences survey
        food_preferences_df = pd.read_csv(os.path.join(self.data_path, 'Food_Preference.csv'))
        
        # Dataset 8: food.csv - Food descriptions
        food_descriptions_df = pd.read_csv(os.path.join(self.data_path, 'food.csv'))
        
        # Dataset 9: nutrition.csv - Detailed dish information
        dishes_detailed_df = pd.read_csv(os.path.join(self.data_path, 'nutrition.csv'))
        
        # Dataset 10: Food_Recipe.csv - Recipe details with ingredients
        recipe_details_df = pd.read_csv(os.path.join(self.data_path, 'Food_Recipe.csv'))
        
        print("Processing user profiles...")
        # Process user profiles (combining datasets 3, 4, 6, 7)
        self.user_profiles = self._process_user_profiles(
            user_profiles_df, patient_data_df, diet_recommendations_df, food_preferences_df
        )
        
        print("Processing recipe data...")
        # Process recipe data (combining datasets 1, 8, 9, 10)
        self.recipe_features = self._process_recipe_data(
            diet_recipes_df, food_descriptions_df, dishes_detailed_df, recipe_details_df, food_calories_df
        )
        
        print("Creating user-recipe interactions...")
        # Create user-recipe interaction data
        self.interactions_df = self._create_user_recipe_interactions(
            self.user_profiles, self.recipe_features, user_consumption_df
        )
        
        print("Preparing training data...")
        # Prepare data for training
        self.train_data, self.test_data = self._prepare_training_data()
        
        return self.train_data, self.test_data
    
    def _process_user_profiles(self, health_df, medical_df, diet_df, preferences_df):
        """Process and combine user profile data from actual datasets"""
        users_data = []
        
        # Process detailed_meals_macros_.csv as primary source
        for i, row in health_df.iterrows():
            # Get corresponding data from other datasets (using modulo for cycling)
            medical_idx = i % len(medical_df) if len(medical_df) > 0 else 0
            diet_idx = i % len(diet_df) if len(diet_df) > 0 else 0
            pref_idx = i % len(preferences_df) if len(preferences_df) > 0 else 0
            
            medical_row = medical_df.iloc[medical_idx] if len(medical_df) > 0 else {}
            diet_row = diet_df.iloc[diet_idx] if len(diet_df) > 0 else {}
            pref_row = preferences_df.iloc[pref_idx] if len(preferences_df) > 0 else {}
            
            user_profile = {
                'user_id': f'user_{i}',
                'age': row.get('Ages', 30),
                'gender': row.get('Gender', 'Male'),
                'height': row.get('Height', 170),
                'weight': row.get('Weight', 70),
                'activity_level': row.get('Activity Level', 'Moderately Active'),
                'dietary_preference': row.get('Dietary Preference', 'Omnivore'),
                'daily_calorie_target': row.get('Daily Calorie Target', 2000),
                'protein_target': row.get('Protein', 100),
                'carb_target': row.get('Carbohydrates', 250),
                'fat_target': row.get('Fat', 65),
                'disease': row.get('Disease', 'None'),
                'bmi': medical_row.get('BMI', 22.5),
                'allergies': medical_row.get('Allergies', 'None'),
                'preferred_cuisine': medical_row.get('Preferred_Cuisine', 'International'),
                'nationality': pref_row.get('Nationality', 'International'),
                'breakfast_suggestion': row.get('Breakfast Suggestion', 'Oatmeal'),
                'lunch_suggestion': row.get('Lunch Suggestion', 'Salad'),
                'dinner_suggestion': row.get('Dinner Suggestion', 'Salmon'),
                'snack_suggestion': row.get('Snack Suggestion', 'Yogurt')
            }
            users_data.append(user_profile)
        
        users_df = pd.DataFrame(users_data)
        
        # Handle missing values
        users_df = users_df.fillna({
            'bmi': 22.5,
            'allergies': 'None',
            'preferred_cuisine': 'International',
            'nationality': 'International'
        })
        
        # Encode categorical features
        users_df['gender_encoded'] = LabelEncoder().fit_transform(users_df['gender'])
        users_df['activity_encoded'] = self.activity_encoder.fit_transform(users_df['activity_level'])
        users_df['diet_pref_encoded'] = LabelEncoder().fit_transform(users_df['dietary_preference'])
        users_df['cuisine_pref_encoded'] = LabelEncoder().fit_transform(users_df['preferred_cuisine'])
        
        # Calculate BMI if not available
        users_df['calculated_bmi'] = users_df['weight'] / ((users_df['height']/100) ** 2)
        users_df['bmi'] = users_df['bmi'].fillna(users_df['calculated_bmi'])
        
        return users_df
    
    def _process_recipe_data(self, diet_recipes_df, descriptions_df, dishes_df, recipe_details_df, calories_df):
        """Process and combine recipe data from actual datasets"""
        all_recipes = []
        
        # Process All_Diets.csv (dataset 1)
        for i, row in diet_recipes_df.iterrows():
            recipe = {
                'recipe_id': f'diet_{i}',
                'name': row.get('Recipe_name', f'Recipe_{i}'),
                'diet_type': row.get('Diet_type', 'general'),
                'cuisine': row.get('Cuisine_type', 'international'),
                'protein': max(0, row.get('Protein(g)', 10)),
                'carbs': max(0, row.get('Carbs(g)', 30)),
                'fat': max(0, row.get('Fat(g)', 10)),
                'calories': 0,  # Will calculate
                'meal_type': 'main',
                'veg_non_veg': 'veg' if row.get('Diet_type') in ['vegan', 'vegetarian'] else 'non-veg',
                'allergens': 'None',
                'source': 'diet_recipes'
            }
            recipe['calories'] = recipe['protein'] * 4 + recipe['carbs'] * 4 + recipe['fat'] * 9
            all_recipes.append(recipe)
        
        # Process nutrition.csv (dataset 9)
        for i, row in dishes_df.iterrows():
            recipe = {
                'recipe_id': f'dish_{i}',
                'name': row.get('Dish Name', f'Dish_{i}'),
                'diet_type': str(row.get('Diet', 'general')).lower(),
                'cuisine': row.get('Cuisine', 'international'),
                'protein': max(0, row.get('Protein (g)', 10)),
                'carbs': max(0, row.get('Carbohydrates (g)', 30)),
                'fat': max(0, row.get('Fat (g)', 10)),
                'calories': max(0, row.get('Calories (kcal)', 200)),
                'meal_type': str(row.get('Meal Type', 'main')).lower(),
                'veg_non_veg': 'veg' if 'veg' in str(row.get('Diet', '')).lower() else 'non-veg',
                'allergens': row.get('Allergens', 'None'),
                'source': 'nutrition'
            }
            all_recipes.append(recipe)
        
        # Process Food_Recipe.csv (dataset 10)
        for i, row in recipe_details_df.iterrows():
            # Estimate nutrition from ingredients or use defaults
            protein_est = 15 + np.random.normal(0, 5)
            carbs_est = 35 + np.random.normal(0, 10)
            fat_est = 12 + np.random.normal(0, 4)
            
            recipe = {
                'recipe_id': f'recipe_{i}',
                'name': row.get('name', f'Recipe_{i}'),
                'diet_type': str(row.get('diet', 'general')).lower(),
                'cuisine': row.get('cuisine', 'international'),
                'protein': max(0, protein_est),
                'carbs': max(0, carbs_est),
                'fat': max(0, fat_est),
                'calories': 0,  # Will calculate
                'meal_type': str(row.get('course', 'main')).lower(),
                'veg_non_veg': 'veg' if 'vegetarian' in str(row.get('diet', '')).lower() else 'non-veg',
                'allergens': 'None',
                'prep_time': row.get('prep_time (in mins)', 30),
                'cook_time': row.get('cook_time (in mins)', 30),
                'source': 'recipes'
            }
            recipe['calories'] = recipe['protein'] * 4 + recipe['carbs'] * 4 + recipe['fat'] * 9
            all_recipes.append(recipe)
        
        # Process food.csv (dataset 8) - simple foods
        for i, row in descriptions_df.iterrows():
            # Estimate nutrition based on food type
            protein_base = 8 if row.get('C_Type') == 'Protein' else 5
            carbs_base = 25 if 'grain' in str(row.get('Describe', '')).lower() else 15
            fat_base = 5
            
            recipe = {
                'recipe_id': f'food_{i}',
                'name': row.get('Name', f'Food_{i}'),
                'diet_type': 'general',
                'cuisine': 'international',
                'protein': max(0, protein_base + np.random.normal(0, 2)),
                'carbs': max(0, carbs_base + np.random.normal(0, 5)),
                'fat': max(0, fat_base + np.random.normal(0, 2)),
                'calories': 0,  # Will calculate
                'meal_type': 'main',
                'veg_non_veg': row.get('Veg_Non', 'veg'),
                'allergens': 'None',
                'source': 'food_descriptions'
            }
            recipe['calories'] = recipe['protein'] * 4 + recipe['carbs'] * 4 + recipe['fat'] * 9
            all_recipes.append(recipe)
        
        recipes_df = pd.DataFrame(all_recipes)
        
        # Clean and standardize data
        recipes_df = recipes_df.drop_duplicates(subset=['name'])
        recipes_df = recipes_df.reset_index(drop=True)
        recipes_df['recipe_id'] = [f'recipe_{i}' for i in range(len(recipes_df))]
        
        # Clean diet_type values
        recipes_df['diet_type'] = recipes_df['diet_type'].str.lower().str.strip()
        recipes_df['diet_type'] = recipes_df['diet_type'].replace({'veg': 'vegetarian', 'non-veg': 'omnivore'})
        
        # Clean cuisine values
        recipes_df['cuisine'] = recipes_df['cuisine'].str.lower().str.strip()
        
        # Clean meal_type values
        recipes_df['meal_type'] = recipes_df['meal_type'].str.lower().str.strip()
        recipes_df['meal_type'] = recipes_df['meal_type'].replace({'appetizer': 'snack', 'course': 'main'})
        
        # Encode categorical features
        recipes_df['diet_type_encoded'] = self.diet_encoder.fit_transform(recipes_df['diet_type'])
        recipes_df['cuisine_encoded'] = self.cuisine_encoder.fit_transform(recipes_df['cuisine'])
        recipes_df['meal_type_encoded'] = LabelEncoder().fit_transform(recipes_df['meal_type'])
        recipes_df['veg_encoded'] = LabelEncoder().fit_transform(recipes_df['veg_non_veg'])
        
        # Normalize nutritional values
        nutrition_cols = ['protein', 'carbs', 'fat', 'calories']
        recipes_df[nutrition_cols] = self.nutrition_scaler.fit_transform(recipes_df[nutrition_cols])
        
        return recipes_df
    
    def _create_user_recipe_interactions(self, users_df, recipes_df, consumption_df=None):
        """Create user-recipe interaction data with health-based scoring"""
        interactions = []
        
        print(f"Creating interactions for {len(users_df)} users and {len(recipes_df)} recipes...")
        
        for _, user in users_df.iterrows():
            # Generate interactions for each user based on their preferences
            num_interactions = np.random.randint(15, 30)  # Each user rates 15-30 recipes
            recipe_indices = np.random.choice(len(recipes_df), min(num_interactions, len(recipes_df)), replace=False)
            
            for recipe_idx in recipe_indices:
                recipe = recipes_df.iloc[recipe_idx]
                
                # Calculate compatibility score based on health factors
                compatibility_score = self._calculate_health_compatibility(user, recipe)
                
                # Convert to rating (1-5 scale)
                rating = max(1, min(5, int(compatibility_score * 5)))
                
                interaction = {
                    'user_id': user['user_id'],
                    'recipe_id': recipe['recipe_id'],
                    'rating': rating,
                    'compatibility_score': compatibility_score
                }
                interactions.append(interaction)
        
        interactions_df = pd.DataFrame(interactions)
        
        # Encode IDs
        interactions_df['user_id_encoded'] = self.user_encoder.fit_transform(interactions_df['user_id'])
        interactions_df['recipe_id_encoded'] = self.recipe_encoder.fit_transform(interactions_df['recipe_id'])
        
        return interactions_df
    
    def _calculate_health_compatibility(self, user, recipe):
        """Calculate how compatible a recipe is with user's health profile"""
        score = 0.5  # Base score
        
        # Diet type compatibility
        user_diet = str(user['dietary_preference']).lower() if pd.notna(user['dietary_preference']) else 'omnivore'
        recipe_diet = str(recipe['diet_type']).lower() if pd.notna(recipe['diet_type']) else 'general'
        
        if user_diet == 'vegetarian' and recipe['veg_non_veg'] == 'veg':
            score += 0.3
        elif user_diet == 'vegan' and recipe_diet == 'vegan':
            score += 0.3
        elif user_diet == 'omnivore':
            score += 0.1
        elif user_diet.lower() == recipe_diet.lower():
            score += 0.2
        
        # Cuisine preference
        user_cuisine = str(user['preferred_cuisine']).lower() if pd.notna(user['preferred_cuisine']) else 'international'
        recipe_cuisine = str(recipe['cuisine']).lower() if pd.notna(recipe['cuisine']) else 'international'
        
        if user_cuisine == recipe_cuisine:
            score += 0.2
        
        # Calorie compatibility (denormalize for comparison)
        user_calorie_target = user['daily_calorie_target'] / 3  # Assuming 3 meals per day
        recipe_calories = recipe['calories'] * 1000  # Denormalize (assuming max 1000 cal for normalization)
        calorie_diff = abs(user_calorie_target - recipe_calories) / max(user_calorie_target, 1)
        score += 0.2 * (1 - min(calorie_diff, 1))
        
        # Nutritional balance based on health conditions
        protein_ratio = recipe['protein'] / (recipe['protein'] + recipe['carbs'] + recipe['fat'] + 0.001)
        
        user_disease = str(user['disease']).lower() if pd.notna(user['disease']) else 'none'
        
        if user_disease == 'weight gain' and protein_ratio > 0.3:
            score += 0.2
        elif user_disease == 'diabetes' and recipe['carbs'] < 0.3:  # Lower carbs for diabetes
            score += 0.2
        elif user_disease == 'heart disease' and recipe['fat'] < 0.25:  # Lower fat for heart disease
            score += 0.2
        
        # Age factor
        if user['age'] > 50 and recipe_calories < user_calorie_target * 0.8:
            score += 0.1
        
        # Activity level factor
        user_activity = str(user['activity_level']).lower() if pd.notna(user['activity_level']) else 'moderate'
        if 'very active' in user_activity and protein_ratio > 0.25:
            score += 0.1
        
        return min(1.0, max(0.1, score))  # Keep score between 0.1 and 1.0
    
    def _prepare_training_data(self):
        """Prepare data for model training"""
        # Merge user and recipe features with interactions
        train_data = self.interactions_df.merge(
            self.user_profiles[['user_id', 'age', 'gender_encoded', 'bmi', 'activity_encoded', 
                              'diet_pref_encoded', 'daily_calorie_target']],
            on='user_id'
        ).merge(
            self.recipe_features[['recipe_id', 'diet_type_encoded', 'cuisine_encoded', 
                                'meal_type_encoded', 'protein', 'carbs', 'fat', 'calories']],
            on='recipe_id'
        )
        
        # Split data
        train_df, test_df = train_test_split(train_data, test_size=0.2, random_state=42)
        
        return train_df, test_df
    
    def build_model(self, embedding_dim=64):
        """Build the neural collaborative filtering model with health features"""
        n_users = len(self.user_encoder.classes_)
        n_recipes = len(self.recipe_encoder.classes_)
        
        # User inputs
        user_input = Input(shape=(), name='user_id')
        user_embedding = Embedding(n_users, embedding_dim, name='user_embedding')(user_input)
        user_vec = Flatten(name='user_flatten')(user_embedding)
        
        # Recipe inputs
        recipe_input = Input(shape=(), name='recipe_id')
        recipe_embedding = Embedding(n_recipes, embedding_dim, name='recipe_embedding')(recipe_input)
        recipe_vec = Flatten(name='recipe_flatten')(recipe_embedding)
        
        # User features
        user_age = Input(shape=(1,), name='user_age')
        user_bmi = Input(shape=(1,), name='user_bmi')
        user_calories = Input(shape=(1,), name='user_calories')
        user_activity = Input(shape=(1,), name='user_activity')
        user_diet_pref = Input(shape=(1,), name='user_diet_pref')
        
        # Recipe features
        recipe_protein = Input(shape=(1,), name='recipe_protein')
        recipe_carbs = Input(shape=(1,), name='recipe_carbs')
        recipe_fat = Input(shape=(1,), name='recipe_fat')
        recipe_calories = Input(shape=(1,), name='recipe_calories')
        recipe_diet_type = Input(shape=(1,), name='recipe_diet_type')
        
        # Combine all features
        combined = Concatenate()([
            user_vec, recipe_vec,
            user_age, user_bmi, user_calories, user_activity, user_diet_pref,
            recipe_protein, recipe_carbs, recipe_fat, recipe_calories, recipe_diet_type
        ])
        
        # Neural network layers
        dense1 = Dense(256, activation='relu')(combined)
        dropout1 = Dropout(0.3)(dense1)
        batch_norm1 = BatchNormalization()(dropout1)
        
        dense2 = Dense(128, activation='relu')(batch_norm1)
        dropout2 = Dropout(0.3)(dense2)
        batch_norm2 = BatchNormalization()(dropout2)
        
        dense3 = Dense(64, activation='relu')(batch_norm2)
        dropout3 = Dropout(0.2)(dense3)
        
        output = Dense(1, activation='sigmoid', name='rating')(dropout3)
        
        # Create model
        self.model = Model(
            inputs=[user_input, recipe_input, user_age, user_bmi, user_calories, 
                   user_activity, user_diet_pref, recipe_protein, recipe_carbs, 
                   recipe_fat, recipe_calories, recipe_diet_type],
            outputs=output
        )
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print("Model Architecture:")
        self.model.summary()
        
        return self.model
    
    def train(self, epochs=50, batch_size=64):
        """Train the model"""
        # Prepare training inputs
        train_inputs = [
            self.train_data['user_id_encoded'].values,
            self.train_data['recipe_id_encoded'].values,
            self.train_data['age'].values.reshape(-1, 1),
            self.train_data['bmi'].values.reshape(-1, 1),
            self.train_data['daily_calorie_target'].values.reshape(-1, 1),
            self.train_data['activity_encoded'].values.reshape(-1, 1),
            self.train_data['diet_pref_encoded'].values.reshape(-1, 1),
            self.train_data['protein'].values.reshape(-1, 1),
            self.train_data['carbs'].values.reshape(-1, 1),
            self.train_data['fat'].values.reshape(-1, 1),
            self.train_data['calories'].values.reshape(-1, 1),
            self.train_data['diet_type_encoded'].values.reshape(-1, 1)
        ]
        
        train_targets = self.train_data['rating'].values / 5.0  # Normalize to 0-1
        
        # Prepare test inputs
        test_inputs = [
            self.test_data['user_id_encoded'].values,
            self.test_data['recipe_id_encoded'].values,
            self.test_data['age'].values.reshape(-1, 1),
            self.test_data['bmi'].values.reshape(-1, 1),
            self.test_data['daily_calorie_target'].values.reshape(-1, 1),
            self.test_data['activity_encoded'].values.reshape(-1, 1),
            self.test_data['diet_pref_encoded'].values.reshape(-1, 1),
            self.test_data['protein'].values.reshape(-1, 1),
            self.test_data['carbs'].values.reshape(-1, 1),
            self.test_data['fat'].values.reshape(-1, 1),
            self.test_data['calories'].values.reshape(-1, 1),
            self.test_data['diet_type_encoded'].values.reshape(-1, 1)
        ]
        
        test_targets = self.test_data['rating'].values / 5.0
        
        # Train model
        history = self.model.fit(
            train_inputs, train_targets,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(test_inputs, test_targets),
            verbose=1
        )
        
        # Save model
        os.makedirs('trained_models', exist_ok=True)
        self.model.save('trained_models/personalized_meal_recommender.keras')
        
        # Plot training history
        self._plot_training_history(history)
        
        return history
    
    def _plot_training_history(self, history):
        """Plot training history"""
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
        plt.savefig('trained_models/training_history.png')
        plt.show()
    
    def recommend_meals(self, user_id, top_n=10, meal_type=None, diet_restriction=None):
        """Generate personalized meal recommendations"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get user profile
        user_profile = self.user_profiles[self.user_profiles['user_id'] == user_id]
        if user_profile.empty:
            raise ValueError(f"User {user_id} not found!")
        
        user_profile = user_profile.iloc[0]
        user_encoded = self.user_encoder.transform([user_id])[0]
        
        # Get all recipes (filter by meal_type and diet_restriction if specified)
        recipes_to_consider = self.recipe_features.copy()
        
        if meal_type:
            recipes_to_consider = recipes_to_consider[recipes_to_consider['meal_type'] == meal_type.lower()]
        
        if diet_restriction:
            recipes_to_consider = recipes_to_consider[recipes_to_consider['diet_type'] == diet_restriction.lower()]
        
        if len(recipes_to_consider) == 0:
            return []
        
        # Prepare prediction inputs
        n_recipes = len(recipes_to_consider)
        
        pred_inputs = [
            np.full(n_recipes, user_encoded),
            recipes_to_consider['recipe_id_encoded'].values,
            np.full((n_recipes, 1), user_profile['age']),
            np.full((n_recipes, 1), user_profile['bmi']),
            np.full((n_recipes, 1), user_profile['daily_calorie_target']),
            np.full((n_recipes, 1), user_profile['activity_encoded']),
            np.full((n_recipes, 1), user_profile['diet_pref_encoded']),
            recipes_to_consider['protein'].values.reshape(-1, 1),
            recipes_to_consider['carbs'].values.reshape(-1, 1),
            recipes_to_consider['fat'].values.reshape(-1, 1),
            recipes_to_consider['calories'].values.reshape(-1, 1),
            recipes_to_consider['diet_type_encoded'].values.reshape(-1, 1)
        ]
        
        # Predict ratings
        predictions = self.model.predict(pred_inputs)
        predictions = predictions.flatten() * 5  # Denormalize to 1-5 scale
        
        # Get top recommendations
        recipe_scores = list(zip(recipes_to_consider['recipe_id'].values, 
                               recipes_to_consider['name'].values,
                               recipes_to_consider['diet_type'].values,
                               recipes_to_consider['cuisine'].values,
                               predictions))
        
        # Sort by predicted rating
        recipe_scores.sort(key=lambda x: x[4], reverse=True)
        
        # Format recommendations
        recommendations = []
        for i in range(min(top_n, len(recipe_scores))):
            recipe_id, name, diet_type, cuisine, score = recipe_scores[i]
            recipe_details = recipes_to_consider[recipes_to_consider['recipe_id'] == recipe_id].iloc[0]
            
            recommendation = {
                'recipe_id': recipe_id,
                'name': name,
                'predicted_rating': score,
                'diet_type': diet_type,
                'cuisine': cuisine,
                'calories': recipe_details['calories'] * 1000,  # Denormalize
                'protein': recipe_details['protein'] * 100,
                'carbs': recipe_details['carbs'] * 100,
                'fat': recipe_details['fat'] * 100,
                'health_compatibility': self._calculate_health_compatibility(user_profile, recipe_details)
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def get_user_health_insights(self, user_id):
        """Get health insights and dietary recommendations for a user"""
        user_profile = self.user_profiles[self.user_profiles['user_id'] == user_id]
        if user_profile.empty:
            return None
        
        user = user_profile.iloc[0]
        insights = {
            'bmi_status': self._get_bmi_status(user['bmi']),
            'calorie_recommendation': user['daily_calorie_target'],
            'protein_target': user['protein_target'],
            'dietary_preference': user['dietary_preference'],
            'health_conditions': user['disease'],
            'recommended_cuisines': [user['preferred_cuisine']],
            'activity_level': user['activity_level']
        }
        
        return insights
    
    def _get_bmi_status(self, bmi):
        """Get BMI status category"""
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi < 25:
            return "Normal"
        elif 25 <= bmi < 30:
            return "Overweight"
        else:
            return "Obese"

def main():
    # Create sample datasets if they don't exist
    if not os.path.exists('datasets/diet_recipes.csv'):
        print("Creating sample datasets...")
        create_sample_datasets()
    
    # Initialize the recommendation system
    print("Initializing Personalized Meal Recommendation System...")
    recommender = PersonalizedMealRecommender(data_path='datasets')
    
    try:
        # Load and preprocess data
        print("Loading and processing health-focused datasets...")
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
                
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec['name']}")
                    print(f"      Rating: {rec['predicted_rating']:.2f}/5 | Diet: {rec['diet_type']}")
                    print(f"      Calories: {rec['calories']:.0f} | Protein: {rec['protein']:.1f}g")
                    print(f"      Health Compatibility: {rec['health_compatibility']:.1%}")
                    print()
                    
            except Exception as e:
                print(f"   No recommendations available: {e}")
        
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
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 