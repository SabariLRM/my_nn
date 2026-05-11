import tensorflow as tf
import cupy as cp
from tqdm import tqdm

print("GPU devices:", cp.cuda.runtime.getDeviceCount())

# ---------- NN (CuPy everywhere) ----------
def init_nn(n_in, hidarr, n_out, dtype=cp.float32, seed=1):
    cp.random.seed(seed)
    sizes = [n_in] + list(hidarr) + [n_out]
    W, b = [], []
    for i in range(1, len(sizes)):
        fan_in = sizes[i-1]
        limit = cp.sqrt(cp.asarray(6.0, dtype=dtype) / cp.asarray(fan_in, dtype=dtype))
        W.append(cp.random.uniform(-limit, limit, size=(sizes[i], sizes[i-1])).astype(dtype))
        b.append(cp.zeros((sizes[i],), dtype=dtype))
    return [W, b]

def relu(x):
    return cp.maximum(x, 0)

def softmax(logits):
    z = logits - cp.max(logits, axis=1, keepdims=True)
    exps = cp.exp(z)
    return exps / cp.sum(exps, axis=1, keepdims=True)

def train_batch(model, X, Y, lr):
    W, b = model
    L = len(W)
    B = X.shape[0]

    # forward
    A_list = [X]
    Z_list = []
    A = X
    for l in range(L):
        Z = A @ W[l].T + b[l]          # (B, out)
        Z_list.append(Z)
        A = relu(Z) if l != L - 1 else Z
        A_list.append(A)

    P = softmax(A_list[-1])            # (B, n_out)

    # backward (mean over batch)
    dZ = (P - Y) / B                   # (B, out)

    for l in range(L - 1, -1, -1):
        A_prev = A_list[l]             # (B, in)
        dW = dZ.T @ A_prev             # (out, in)
        db = cp.sum(dZ, axis=0)        # (out,)

        W[l] -= lr * dW
        b[l] -= lr * db

        if l > 0:
            dA_prev = dZ @ W[l]        # (B, in)
            dZ = dA_prev * (Z_list[l - 1] > 0)

    return [W, b]

def predict_proba(model, X):
    W, b = model
    A = X
    for l in range(len(W)):
        Z = A @ W[l].T + b[l]
        A = relu(Z) if l != len(W) - 1 else Z
    return softmax(A)

# ---------- Load MNIST (CPU) -> move to GPU ----------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 784).astype("float32") / 255.0

X_train = cp.asarray(x_train)                 # (60000, 784)
X_test  = cp.asarray(x_test)                  # (10000, 784)

y_train_cp = cp.asarray(y_train, dtype=cp.int32)
y_test_cp  = cp.asarray(y_test,  dtype=cp.int32)

Y_train = cp.eye(10, dtype=cp.float32)[y_train_cp]   # (60000, 10)

# ---------- Train on ALL 60000 ----------
model = init_nn(784, [128, 64], 10, seed=1)
lr = 0.001
batch_size = 2
epochs = 50  # set higher if you want better accuracy

for ep in range(epochs):
    idx = cp.random.permutation(X_train.shape[0])
    Xs = X_train[idx]
    Ys = Y_train[idx]

    for i in tqdm(range(0, Xs.shape[0], batch_size), desc=f"train epoch {ep+1}"):
        xb = Xs[i:i+batch_size]
        yb = Ys[i:i+batch_size]
        model = train_batch(model, xb, yb, lr)

# ---------- Test on ALL 10000 ----------
correct = 0
total = X_test.shape[0]

for i in tqdm(range(0, total, batch_size), desc="test"):
    xb = X_test[i:i+batch_size]
    yb = y_test_cp[i:i+batch_size]
    probs = predict_proba(model, xb)
    pred = cp.argmax(probs, axis=1)
    correct += int(cp.sum(pred == yb).item())

print("Accuracy:", correct / total)
