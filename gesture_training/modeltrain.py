import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle
import os

CSV_FILE_NAME = "hand_dataset.csv"

if not os.path.exists(CSV_FILE_NAME):
    print(f"ERROR: Could not find '{CSV_FILE_NAME}'.")
    print("Please run 'extract_coordinates.py' first.")
    exit()

print("Loading localized dataset...")
df = pd.read_csv(CSV_FILE_NAME)

X = df.drop('label', axis=1).values  # Normalized 63 coordinates
y_raw = df['label'].values           # String names

# Encode string labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
NUM_CLASSES = len(label_encoder.classes_)

# Save class mappings
with open('label_classes.pkl', 'wb') as f:
    pickle.dump(label_encoder.classes_, f)
print(f"Target classes identified: {label_encoder.classes_}")

# --- CRITICAL BUG FIX: ONE-HOT ENCODING FOR MULTI-LABEL SIGMOID ---
# To train independent Sigmoid neurons, we convert our integer targets to one-hot vectors
y_one_hot = tf.keras.utils.to_categorical(y_encoded, num_classes=NUM_CLASSES)

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Build deep feed-forward coordinate model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    
    tf.keras.layers.Dense(128, activation='relu'), #relu is for hidden layers.
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    
    # Use 'sigmoid' instead of 'softmax' so output scores are independent (0 to 1)
    tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid')
])

# Use binary_crossentropy because each class output behaves as an independent binary decision
model.compile(
    optimizer='adam', #best optimizer
    loss='binary_crossentropy', #used for only binary classification where targets are between 0 and 1.
    metrics=['accuracy']
)

history = model.fit(
    X_train, 
    y_train, 
    epochs=65, 
    batch_size=16, 
    validation_data=(X_test, y_test)
)

print("\nEvaluating model performance...")
val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Model Accuracy: {val_acc*100:.2f}%")

# Generate performance report
predictions = model.predict(X_test)
pred_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

print("\n--- Model Performance Report ---")
print(classification_report(true_classes, pred_classes, target_names=label_encoder.classes_))

# Save models
keras_filename = 'gesture_coordinate_model.keras'
model.save(keras_filename)
print(f"Saved robust Keras model: '{keras_filename}'")
