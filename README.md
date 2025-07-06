# Deep-Learning
Artificial Neural Network, Feed-Forward Neural Network, Back Propagation
# 🧠 Artificial Neural Networks (ANN), Feed Forward Neural Networks (FFNN), and Backpropagation (BP)

## 📌 Overview

This repository provides an understanding and implementation of:
- **Artificial Neural Networks (ANN)**
- **Feed Forward Neural Networks (FFNN)**
- **Backpropagation (BP)**

It includes concepts, architectures, and sample Python implementations using libraries like NumPy and TensorFlow/Keras.

---

## 🔍 What is ANN?

An **Artificial Neural Network (ANN)** is a computational model inspired by the human brain. It consists of layers of interconnected nodes (neurons), where each connection has a weight. ANN is used in tasks like image classification, speech recognition, and forecasting.

### Components:
- **Input Layer**: Receives input features.
- **Hidden Layers**: Process input with learned weights and activation functions.
- **Output Layer**: Produces the final result or prediction.

---

## 🚀 Feed Forward Neural Network (FFNN)

A **Feed Forward Neural Network (FFNN)** is the simplest type of ANN where the information flows in one direction — from input to output — with no cycles or loops.

### Key Features:
- No feedback connections
- Can have one or multiple hidden layers
- Suitable for regression and classification problems


---

## 🔁 Backpropagation (BP)

**Backpropagation** is the learning algorithm used to train ANNs, especially FFNNs.

### How It Works:
1. **Forward Pass**: Input is passed through the network to get the output.
2. **Loss Calculation**: The output is compared to the actual label using a loss function.
3. **Backward Pass**: The loss is propagated back through the network using the chain rule of calculus.
4. **Weight Update**: Weights are adjusted using gradient descent to minimize the error.

### Objective:
To minimize the error between the predicted and actual outputs by adjusting weights.

---

## 🧪 Sample Code (Keras)

python
from keras.models import Sequential
from keras.layers import Dense

# Define FFNN model
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))  # Hidden layer
model.add(Dense(3, activation='softmax'))  # Output layer for 3 classes

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)


✅ Use Cases
Image & speech recognition

Natural Language Processing (NLP)

Stock market prediction

Medical diagnosis
