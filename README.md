# my_nn

A high-performance, educational neural network library built completely from scratch using NumPy. 

`my_nn` provides a Keras-like object-oriented API for building and training neural networks. It is designed to be lightweight, avoiding heavy dependencies like TensorFlow or PyTorch, while leveraging highly-optimized C-Extensions (via Cython) for speed and strong protection against reverse engineering.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [Models](#models)
  - [Layers](#layers)
  - [Activations](#activations)
  - [Losses](#losses)
  - [Optimizers](#optimizers)
- [Anti-Reverse Engineering Security](#anti-reverse-engineering-security)

---

## Features

- **Pure NumPy Math**: Built entirely on standard matrix operations without heavy machine learning frameworks.
- **Keras-like API**: Intuitive `Sequential` model structure that makes building networks incredibly easy.
- **C-Extension Compilation**: Python code is compiled via Cython into native machine code `.so` objects, rendering it practically impossible to decompile or reverse engineer.
- **Customizable**: Control the exact size and shape of every layer and activation function.

---

## Installation

You can install `my_nn` directly via `pip` once it is published to PyPI:

```bash
pip install my_nn
```

*(Note: If building from source, ensure you have a C compiler installed, then run `pip install .` to compile the Cython extensions)*

---

## Quick Start

Here is a simple example demonstrating how to build a model to solve the XOR problem:

```python
import numpy as np
from my_nn import Sequential, Dense, ReLU, SoftmaxCrossEntropy
from my_nn.activations import Softmax

# 1. Create Data (XOR problem)
X_train = np.array([[0,0], [0,1], [1,0], [1,1]])
Y_train = np.array([[1, 0], [0, 1], [0, 1], [1, 0]]) # One-hot encoded

# 2. Build the Model
model = Sequential()

# First layer: 2 input neurons (for the 2 XOR inputs), 3 output neurons
model.add(Dense(input_size=2, output_size=3))
model.add(ReLU())

# Second layer: 3 input neurons (must match previous layer), 2 output neurons
model.add(Dense(input_size=3, output_size=2))

# Note: We output raw logits directly to the loss function for numerical stability.

# 3. Compile and Train
loss = SoftmaxCrossEntropy()
model.use(loss, loss.prime)
model.fit(X_train, Y_train, epochs=1000, learning_rate=0.1, batch_size=4)

# 4. Predict
predictions = model.predict_batch(X_train)

# Apply softmax to raw logits to get final probabilities
probs = Softmax().forward(predictions)
print(probs)
```

---

## API Reference

### Models

#### `Sequential()`
The core container for stacking layers.
- `add(layer)`: Appends a layer (Dense or Activation) to the network.
- `use(loss, loss_prime)`: Sets the loss function and its derivative.
- `fit(x_train, y_train, epochs, learning_rate, batch_size, verbose)`: Trains the model.
- `predict(input_data)`: Runs a forward pass on individual samples.
- `predict_batch(input_data)`: Runs a vectorized forward pass on a batch of samples.

### Layers

#### `Dense(input_size, output_size, seed=None)`
A standard fully-connected neural network layer.
- **`input_size`**: The number of input neurons. This must match the `output_size` of the previous layer, or the feature dimension of your dataset for the first layer.
- **`output_size`**: The number of output neurons.
- Uses **He Initialization** automatically to prevent vanishing or exploding gradients.

### Activations

You can append activation functions directly to your `Sequential` model:
- `ReLU()`: Rectified Linear Unit. The standard for hidden layers.
- `Sigmoid()`: Squashes outputs to a `[0, 1]` range.
- `Tanh()`: Squashes outputs to a `[-1, 1]` range.
- `Softmax()`: Converts a vector of logits into a probability distribution.

### Losses

Loss functions are used to calculate the network's error.
- `mse(y_true, y_pred)` & `mse_prime(y_true, y_pred)`: Mean Squared Error.
- `categorical_crossentropy(y_true, y_pred)` & `categorical_crossentropy_prime(y_true, y_pred)`: Cross-Entropy loss.
- `SoftmaxCrossEntropy()`: A highly recommended class that combines Softmax and Cross-Entropy for optimal numerical stability during backpropagation.

### Optimizers

- `SGD(learning_rate)`: Stochastic Gradient Descent (used automatically by `Sequential.fit()`).

---

## Anti-Reverse Engineering Security

This library distributes native C-Extensions instead of Python bytecode. All core logic (layers, backpropagation, and loss functions) is compiled via `Cython` into `.so` shared objects during the build process.

This design choice ensures that:
1. **Performance** is maximized by removing Python interpreter overhead.
2. **Reverse Engineering** is virtually impossible using standard Python decompilers (like `uncompyle6`, `pycdc`, etc.), as the distributed code is stripped of its Python AST and represented as optimized machine code.
