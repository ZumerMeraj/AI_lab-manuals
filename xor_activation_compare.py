import numpy as np
import matplotlib.pyplot as plt
import time

# ======================
#   XOR DATASET
# ======================
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]], dtype=float)

y = np.array([[0],[1],[1],[0]], dtype=float)

np.random.seed(1)

# ======================
# ACTIVATION FUNCTIONS
# ======================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return (x > 0).astype(float)

# ======================
# LOSS FUNCTION
# ======================
def bce_loss(y_true, y_pred, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1-eps)
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(y_pred))

# ======================
# TRAINING FUNCTION
# ======================
def train_network(act, dact, hidden=4, lr=0.1, max_epochs=5000):
    # weights
    W1 = np.random.randn(2, hidden)*0.5
    b1 = np.zeros((1, hidden))
    W2 = np.random.randn(hidden, 1)*0.5
    b2 = np.zeros((1,1))

    losses = []
    accs = []

    t0 = time.perf_counter()
    perfect_count = 0

    for epoch in range(1, max_epochs + 1):

        # Forward pass
        z1 = X @ W1 + b1
        a1 = act(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)

        # Loss + Accuracy
        loss = bce_loss(y, a2)
        losses.append(loss)

        preds = (a2 > 0.5).astype(float)
        acc = np.mean(preds == y)
        accs.append(acc)

        if acc == 1.0:
            perfect_count += 1
        else:
            perfect_count = 0

        if perfect_count >= 50:
            break

        # Backprop
        m = y.shape[0]
        dz2 = (a2 - y) / m
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * dact(z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    t1 = time.perf_counter()
    return losses, accs, epoch, t1 - t0


# ======================
# RUN EXPERIMENT
# ======================
activations = {
    "sigmoid": (sigmoid, dsigmoid),
    "tanh": (tanh, dtanh),
    "relu": (relu, drelu)
}

results = {}

for name, (act, dact) in activations.items():
    print(f"Training with {name} activation...")
    losses, accs, epochs, tsec = train_network(act, dact)
    results[name] = {
        "losses": losses,
        "accs": accs,
        "epochs": epochs,
        "time": tsec
    }
    print(f"{name} → epochs: {epochs}, final acc: {accs[-1]}, time: {tsec:.4f}s")


# ======================
# PLOTS
# ======================

# LOSS PLOT
plt.figure(figsize=(10,5))
for name, r in results.items():
    plt.plot(r["losses"], label=f"{name} (ep={r['epochs']})")
plt.title("Loss vs Epoch (XOR)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ACCURACY PLOT
plt.figure(figsize=(10,5))
for name, r in results.items():
    plt.plot(r["accs"], label=f"{name} (ep={r['epochs']})")
plt.title("Accuracy vs Epoch (XOR)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
