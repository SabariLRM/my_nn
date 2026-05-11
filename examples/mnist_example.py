import os
import sys

# Add the parent directory to the path so we can import tinynn without installing it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from neural_scratch import Sequential, Dense, ReLU, Softmax, SoftmaxCrossEntropy, SGD

print("Loading MNIST dataset...")
print("Loading MNIST dataset... (This might take a moment)")
# ---------- Load MNIST (CPU) ----------
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
X = mnist.data.astype("float32") / 255.0
y = mnist.target.astype(np.int32)

x_train, x_test, y_train_np, y_test_np = train_test_split(X, y, test_size=10000, random_state=42)

# Using numpy
X_train = np.asarray(x_train)
X_test  = np.asarray(x_test)

Y_train = np.eye(10, dtype=np.float32)[y_train_np]   # (60000, 10)

# ---------- Build the Neural Network ----------
print("Building the model...")
model = Sequential()
model.add(Dense(784, 128, seed=1))
model.add(ReLU())
model.add(Dense(128, 64, seed=2))
model.add(ReLU())
model.add(Dense(64, 10, seed=3))
# SoftmaxCrossEntropy is used for stability (it computes softmax internally for the loss)
loss_fn = SoftmaxCrossEntropy()
model.use(loss_fn, loss_fn.prime)

# ---------- Train on ALL 60000 ----------
print("Starting training...")
# Note: we use batch_size=32 and epochs=5 for a reasonable training time on CPU
model.fit(X_train, Y_train, epochs=5, learning_rate=0.01, batch_size=32, verbose=True)

# ---------- Test on ALL 10000 ----------
print("Testing the model...")
# Predict batch returns logits
logits = model.predict_batch(X_test)
# We apply softmax to get probabilities
probs = Softmax().forward(logits)
pred = np.argmax(probs, axis=1)

correct = np.sum(pred == y_test_np)
total = X_test.shape[0]

print(f"Accuracy: {correct / total:.4f} ({correct}/{total})")
