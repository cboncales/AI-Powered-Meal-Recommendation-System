# import joblib
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from keras import layers, models

# class DietRecommender:
#     def __init__(self):
#         self.model = load_model("models/diet_model.h5")
#         preprocessors = joblib.load("models/preprocessors.pkl")
#         self.label_encoders = preprocessors['label_encoders']
#         self.scaler = preprocessors['scaler']
#         self.target_encoder = preprocessors['target_encoder']
    
#     def recommend(self, patient_data):
#         # Convert to DataFrame for easier processing
#         df = pd.DataFrame([patient_data])
        
#         # Preprocess like training data
#         for col, le in self.label_encoders.items():
#             df[col] = le.transform(df[col])
        
#         # Scale features
#         X = self.scaler.transform(df)
        
#         # Make prediction
#         pred = self.model.predict(X)
#         diet_code = np.argmax(pred)
#         diet_name = self.target_encoder.inverse_transform([diet_code])[0]
        
#         return {
#             "diet_recommendation": diet_name,
#             "confidence": float(pred[0][diet_code])
#         }