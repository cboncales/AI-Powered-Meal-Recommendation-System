# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import tensorflow as tf
# from keras import layers, models
# import joblib
# from pathlib import Path
# import matplotlib.pyplot as plt

# # Create models directory if not exists
# Path("models").mkdir(exist_ok=True)

# # Load dataset
# df = pd.read_csv("datasets/diet_recommendations_dataset.csv")

# # Preprocessing
# X = df.drop(columns=["Patient_ID", "Diet_Recommendation"])
# y = df["Diet_Recommendation"]

# # Encode categorical features
# categorical_cols = X.select_dtypes(include="object").columns
# label_encoders = {}
# for col in categorical_cols:
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col])
#     label_encoders[col] = le

# # Scale numeric features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Encode target
# target_encoder = LabelEncoder()
# y_encoded = target_encoder.fit_transform(y)

# # Save preprocessors
# joblib.dump({
#     'label_encoders': label_encoders,
#     'scaler': scaler,
#     'target_encoder': target_encoder
# }, "models/preprocessors.pkl")

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# # Build model
# model = models.Sequential([
#     layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
#     layers.Dropout(0.3),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(len(set(y_encoded)), activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Add callbacks
# callbacks = [
#     tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
#     tf.keras.callbacks.ModelCheckpoint("models/diet_model.h5", save_best_only=True)
# ]

# # Train model
# history = model.fit(X_train, y_train, 
#                    epochs=100, 
#                    batch_size=32, 
#                    validation_split=0.2,
#                    callbacks=callbacks)

# # Evaluate
# loss, acc = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {acc:.2f}")

# # Plot training history
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Accuracy over epochs')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Loss over epochs')
# plt.legend()
# plt.savefig("models/training_history.png")
# plt.close()