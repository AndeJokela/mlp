import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        np.random.seed(5)  # Seed for reproducibility
        
        # He initialization for ReLU layers
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
        # Xavier initialization for sigmoid output layer
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(1. / self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        # Ensure input is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        self.a1 = np.dot(X, self.W1) + self.b1
        self.h1 = self.relu(self.a1)
        self.a2 = np.dot(self.h1, self.W2) + self.b2
        self.output = self.sigmoid(self.a2)
        return self.output

    def backward(self, X, y, lr):
        m = X.shape[0]  # Batch size
        
        # Output layer gradient
        delta2 = 2 * (self.output - y) * self.output * (1 - self.output)
        dW2 = np.dot(self.h1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        
        # Hidden layer gradient
        delta1 = np.dot(delta2, self.W2.T) * (self.h1 > 0)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)
        
        # Parameter updates
        self.W1 -= lr * dW1
        self.W2 -= lr * dW2
        self.b1 -= lr * db1
        self.b2 -= lr * db2

    def MSE(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def train(self, X, y, epochs, lr):
        for _ in range(epochs):
            for i in range(X.shape[0]):
                x_i = X[i].reshape(1, -1)  # Ensure 2D input
                y_i = y[i].reshape(1, -1)
                self.forward(x_i)
                self.backward(x_i, y_i, lr)

    def predict(self, X):
        return self.forward(X)
    
    def accuracy(self, X, y):
        predictions = (self.predict(X) > 0.5).astype(int)
        return np.mean(predictions == y)
    
def test(X_train, y_train):
    print("Training MLP...")
    mlp = MLP(3, 2, 1)
    mlp.train(X_train, y_train, 1000, 0.1)

    print("Testing MLP:")
    for x, target in zip(X_train, y_train):
        pred = mlp.predict(x.reshape(1, -1))
        print(f"data={x}, ground-truth={target[0]}, pred={pred[0][0]:.4f}, rounded={round(pred[0][0])}")

    print(f"Accuracy: {mlp.accuracy(X_train, y_train):.4f}")
    print("-" * 50)

# Example usage
X_train_1 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y_train_1 = np.array([[0], [0], [0], [0], [0], [0], [0], [1]])

X_train_2 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y_train_2 = np.array([[0], [1], [1], [1], [1], [1], [1], [1]])

X_train_3 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y_train_3 = np.array([[0], [0], [0], [1], [0], [1], [0], [0]])

test(X_train_1, y_train_1)
test(X_train_2, y_train_2)
test(X_train_3, y_train_3)