import numpy as np
from sklearn.datasets import load_iris


# Load the Iris dataset
data = load_iris()
X = data.data  # Shape: (150, 4)
y = data.target  # Shape: (150,)

# Split the data into training and testing sets (80% train, 20% test)
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


# One-hot encode the labels
def one_hot_encode(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        one_hot[i, label] = 1
    return one_hot


y_train_one_hot = one_hot_encode(y_train, 3)
y_test_one_hot = one_hot_encode(y_test, 3)


input_size = X.shape[1]  # 4 features
hidden_size1 = 10  # First hidden layer neurons
hidden_size2 = 10  # Second hidden layer neurons
output_size = 3  # 3 classes (iris species)
epochs = 1000
learning_rate = 0.01

# Weights - matrices that transform the input data as it passes from one layer to the next.
# Biases - vectors added to the result of the weighted sum
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2 / input_size)
b1 = np.zeros((1, hidden_size1))
W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2 / hidden_size1)
b2 = np.zeros((1, hidden_size2))
W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2 / hidden_size2)
b3 = np.zeros((1, output_size))


# Sets all negative values in x to 0, and keeps positive values unchanged.
def relu(x):
    return np.maximum(0, x)

# Converts raw output scores (logits) into probabilities that sum to 1.
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Measures the difference between the true distribution (one-hot encoded labels)
# and the predicted probability distribution from the network.
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    return loss



for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(X_train, W1) + b1  # (m, hidden_size1)
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2  # (m, hidden_size2)
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3  # (m, output_size)
    A3 = softmax(Z3)  # (m, output_size) predictions

    # Compute loss
    loss = cross_entropy_loss(y_train_one_hot, A3)

    # Backpropagation
    m = X_train.shape[0]

    dZ3 = A3 - y_train_one_hot  # Difference between prediction and true label
    dW3 = np.dot(A2.T, dZ3) / m  # Gradient for W3
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m  # Gradient for b3

    # Gradients for second hidden layer
    dA2 = np.dot(dZ3, W3.T)  # (m, hidden_size2)
    dZ2 = dA2 * (Z2 > 0)  # ReLU derivative
    dW2 = np.dot(A1.T, dZ2) / m  # (hidden_size1, hidden_size2)
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Gradients for first hidden layer
    dA1 = np.dot(dZ2, W2.T)  # (m, hidden_size1)
    dZ1 = dA1 * (Z1 > 0)  # ReLU derivative
    dW1 = np.dot(X_train.T, dZ1) / m  # (input_size, hidden_size1)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Update parameters
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}: Loss = {loss:.4f}")


def predict(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)
    return np.argmax(A3, axis=1)


predictions = predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
