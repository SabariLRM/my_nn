# neural_scratch

`neural_scratch` is a high-performance, educational neural network library built entirely from scratch using pure NumPy. 

It provides a clean, Keras-like object-oriented API for building and training deep neural networks. To guarantee blazing-fast execution speeds and strong protection against reverse engineering, all the core math and backpropagation logic is compiled directly into native C-extensions via Cython.

## Table of Contents
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Core Components Reference](#core-components-reference)
  - [The Model (`Sequential`)](#the-model-sequential)
  - [Layers (`Dense`)](#layers-dense)
  - [Activations](#activations)
  - [Loss Functions](#loss-functions)
  - [Optimizers](#optimizers)
- [Complete Tutorials](#complete-tutorials)
  - [1. Solving the XOR Problem](#1-solving-the-xor-problem)
  - [2. Multi-class Classification (MNIST)](#2-multi-class-classification-mnist)

---

## Installation

Install directly from PyPI:
```bash
pip install neural_scratch
```
*Note: Because `neural_scratch` distributes raw C source code for cross-platform compatibility, your system will automatically compile it during installation. You must have a C compiler installed on your system.*

---

## How It Works

Building a model in `neural_scratch` follows a simple 4-step pipeline:
1. **Instantiate** a `Sequential` model.
2. **Add** your `Dense` layers and `Activation` layers sequentially.
3. **Configure** the loss function and its derivative using `.use()`.
4. **Train** the model using `.fit()` with your data, epochs, and learning rate.

---

## Core Components Reference

### The Model (`Sequential`)
The `Sequential` class is the container for your network. It passes data forward through your layers during prediction and backwards during training to update weights.

**Methods:**
*   `add(layer)`: Appends a layer (Dense or Activation) to the network.
*   `use(loss, loss_prime)`: Sets the loss function and its derivative for training.
*   `predict(input_data)`: Runs a forward pass on an array of inputs sequentially, returning a list of outputs for each sample.
*   `predict_batch(input_data)`: Runs a vectorized forward pass on an entire batch at once (much faster, recommended for evaluation).
*   `fit(x_train, y_train, epochs, learning_rate, batch_size=None, verbose=True)`: 
    *   `x_train`, `y_train`: Your NumPy training data.
    *   `epochs`: *Integer*. How many times to loop over the entire dataset.
    *   `learning_rate`: *Float*. How big of a step the optimizer takes during gradient descent. Too high causes instability; too low causes slow learning.
    *   `batch_size`: *Integer (Optional)*. If provided, trains using mini-batches (stochastic gradient descent), which is much faster and helps escape local minima. If omitted, uses full-batch gradient descent.
    *   `verbose`: *Boolean*. If True, prints a progress bar (via `tqdm`) and the mean error at each epoch.

### Layers (`Dense`)
The `Dense` layer is a standard fully connected neural network layer where every input neuron is connected to every output neuron.
*   **Syntax:** `Dense(input_size, output_size, seed=None)`
*   **Parameters:**
    *   `input_size`: The number of neurons coming *in* to this layer.
    *   `output_size`: The number of neurons *out* of this layer.
    *   `seed` *(Optional)*: An integer to lock the random number generator for reproducible weight initialization.
*   *Note: `Dense` automatically uses **He Initialization** to optimally scale random weights based on layer size, preventing the vanishing/exploding gradient problem.*

### Activations
Activation functions introduce non-linearity, allowing the network to learn complex patterns instead of just straight lines. You add them directly after a `Dense` layer.
*   `ReLU()`: (Rectified Linear Unit). Outputs the input if positive, otherwise 0. *Best default for hidden layers.*
*   `Sigmoid()`: Squashes outputs to a range between 0 and 1. *Best for binary classification (yes/no).*
*   `Tanh()`: Squashes outputs to a range between -1 and 1. *Usually outperforms Sigmoid in hidden layers.*
*   `Softmax()`: Converts a vector of raw scores (logits) into a probability distribution that sums to 1. *Best for the final layer in multi-class classification.*

### Loss Functions
The loss function measures how "wrong" the network's predictions are. The network's goal is to minimize this value.
*   **Mean Squared Error (MSE)**
    *   *Functions:* `mse`, `mse_prime`
    *   *Use case:* Regression tasks (predicting continuous numbers) or simple binary classification.
*   **Softmax Cross-Entropy**
    *   *Class:* `SoftmaxCrossEntropy()`
    *   *Use case:* Multi-class classification (like identifying digits 0-9). 
    *   *Note:* This class elegantly combines Softmax and Cross-Entropy internally for extreme numerical stability. **Do not add a `Softmax()` activation layer to your model if you are using `SoftmaxCrossEntropy()` as your loss function.** The loss function handles the Softmax calculation mathematically during backpropagation.

### Optimizers
*   `SGD(learning_rate)`: Stochastic Gradient Descent. (Currently, the SGD logic is built directly into the `Sequential.fit` method for efficiency based on your provided `learning_rate` and `batch_size` parameters).

---

## Complete Tutorials

### 1. Solving the XOR Problem
The XOR problem is a classic neural network benchmark. It requires at least one hidden layer with a non-linear activation because the data is not linearly separable.

```python
import numpy as np
from neural_scratch import Sequential, Dense, Tanh, mse, mse_prime

# 1. Prepare Data
# Inputs (4 samples, 2 features)
X_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
# Expected Outputs (4 samples, 1 feature)
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# 2. Build Model
model = Sequential()
model.add(Dense(2, 3))    # Input: 2, Hidden Neurons: 3
model.add(Tanh())         # Hidden Activation
model.add(Dense(3, 1))    # Hidden: 3, Output: 1
model.add(Tanh())         # Output Activation

# 3. Configure & Train
model.use(mse, mse_prime)
model.fit(X_train, y_train, epochs=1000, learning_rate=0.1, verbose=True)

# 4. Predict
predictions = model.predict(X_train)
print("Predictions:\\n", predictions)
```

### 2. Multi-class Classification (MNIST)
For classifying images into multiple distinct categories (like identifying handwritten digits from 0 to 9), we use mini-batches and `SoftmaxCrossEntropy`.

```python
import numpy as np
from neural_scratch import Sequential, Dense, ReLU, SoftmaxCrossEntropy

# Assume X_train is (60000, 784) and Y_train is one-hot encoded (60000, 10)

model = Sequential()
model.add(Dense(784, 128)) # Flattened 28x28 image = 784 inputs
model.add(ReLU())
model.add(Dense(128, 64))
model.add(ReLU())
model.add(Dense(64, 10))   # 10 output classes (digits 0-9)

# Use SoftmaxCrossEntropy for stable multi-class probability loss
loss_fn = SoftmaxCrossEntropy()
model.use(loss_fn, loss_fn.prime)

# Train using mini-batches for significant speedups
model.fit(X_train, Y_train, epochs=10, learning_rate=0.01, batch_size=32)

# To evaluate the model, get raw logits from the network via predict_batch
logits = model.predict_batch(X_test)

# Apply Softmax manually to the output to view final probabilities
from neural_scratch import Softmax
probabilities = Softmax().forward(logits)
predictions = np.argmax(probabilities, axis=1)
```