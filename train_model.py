import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset (assuming it's already balanced ~50/50)
df = pd.read_csv("C:/Users/thegr/Documents/UT Boot Camp/Assignments/project-4/Resources/diabetes_binary_health_indicators_BRFSS2015.csv")
all_features = [col for col in df.columns if col != "Diabetes_binary"]
X = df[all_features]
y = df["Diabetes_binary"]

# Initialize and fit scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define a deeper and wider model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),  # Increased neurons
    keras.layers.Dense(64, activation='relu'),  # Added layer
    keras.layers.Dense(32, activation='relu'),  # More depth
    keras.layers.Dense(16, activation='relu'),  # Gradual reduction
    keras.layers.Dense(1, activation='sigmoid')  # Output layer
])

# Compile with an optimized Adam optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.001)  # Default learning rate, faster convergence
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Custom callback to log training progress
class TrainingLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open("training_log.txt", "a") as f:
            log_message = f"Epoch {epoch+1}/100 - Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}\n"
            f.write(log_message)

# Clear previous log
open("training_log.txt", "w").close()

# Train model with adjusted settings
model.fit(X_scaled, y, 
          epochs=100,  # Increased epochs for more training
          batch_size=64,  # Larger batch size for smoother gradients
          verbose=1, 
          callbacks=[TrainingLogger()])

# Save model and scaler
model.save("diabetes_model.h5")
joblib.dump(scaler, "scaler.pkl")

print("Model, scaler, and training log saved successfully!")