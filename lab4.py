import pandas as pd
import tensorflow as tf
print(tf.__version__)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load the dataset
data = pd.read_csv('pollution_dataset.csv')

# Preprocess the data
X = data.drop(columns=["Air Quality"])
y = data["Air Quality"]

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# First Neural Network: Simple architecture
model_1 = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')  # 4 classes for the target variable
])
model_1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_1.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
loss_1, accuracy_1 = model_1.evaluate(X_test, y_test, verbose=0)

# Second Neural Network: Deeper architecture with dropout
model_2 = Sequential([
    Dense(128, activation='tanh', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='tanh'),
    Dense(32, activation='tanh'),
    Dense(4, activation='softmax')
])
model_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_2.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
loss_2, accuracy_2 = model_2.evaluate(X_test, y_test, verbose=0)

# Third Neural Network: Wider and shallow architecture
model_3 = Sequential([
    Dense(256, activation='elu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='elu'),
    Dense(4, activation='softmax')
])
model_3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_3.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
loss_3, accuracy_3 = model_3.evaluate(X_test, y_test, verbose=0)

# Compare results
print("Model 1: Loss =", loss_1, ", Accuracy =", accuracy_1)
print("Model 2: Loss =", loss_2, ", Accuracy =", accuracy_2)
print("Model 3: Loss =", loss_3, ", Accuracy =", accuracy_3)
