import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

CSV_FILE_NAME = "hand_dataset.csv"

if not os.path.exists(CSV_FILE_NAME):
    print(f"ERROR: Could not find '{CSV_FILE_NAME}'.")
    print("Please run 'extract_coordinates.py' first to build your numerical dataset.")
    exit()

print("Loading localized dataset...")
df = pd.read_csv(CSV_FILE_NAME)

X = df.drop('label', axis=1).values  # Our normalized 63 coordinates
y_raw = df['label'].values           # The text string names

# Translate hand sign names (e.g., 'Orbit_CW') into numbers (0, 1, 2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)

# Save the label encoder to the folder. Your flight control script uses this mapping!
with open('label_classes.pkl', 'wb') as f:
    pickle.dump(label_encoder.classes_, f)
print(f"Target classes identified: {label_encoder.classes_}")

# Partition data: 80% to train, 20% to validate/test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

NUM_CLASSES = len(label_encoder.classes_)

# Build a fast feed-forward neural network for coordinate processing
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(63,)), # High-quality input shape tracking
    
    # Hidden Layer 1 with Dropout to protect against training memorization
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    
    # Hidden Layer 2
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    
    # Output Layer
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training model...")
# Coordinate data is lightweight, allowing us to train for more epochs quickly
history = model.fit(
    X_train, 
    y_train, 
    epochs=60, 
    batch_size=16, 
    validation_data=(X_test, y_test)
)

print("\nEvaluating model performance...")
val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Model Test Accuracy: {val_acc*100:.2f}%")

# Generate advanced diagnostics to make sure classes are well balanced
predictions = np.argmax(model.predict(X_test), axis=1)
print("\n--- Model Performance Report ---")
print(classification_report(y_test, predictions, target_names=label_encoder.classes_))

# Save primary Keras model
keras_filename = 'gesture_coordinate_model.keras'
model.save(keras_filename)
print(f"Saved primary model as: '{keras_filename}'")

# Export highly efficient TFLite model for latency-free flight control
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_filename = 'gesture_coordinate_model.tflite'
with open(tflite_filename, 'wb') as f:
    f.write(tflite_model)
print(f"Saved edge-ready TFLite model as: '{tflite_filename}'")