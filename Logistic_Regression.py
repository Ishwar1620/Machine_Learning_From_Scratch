import numpy as np
class LogisticRegression:

    def __init__(self, learning_rate = 0.01, epochs = 100):
        self.learning_rate =learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.cost_history = []


    def sigmoid(self, z):

        return (1 / (1+np.exp(-z)))

    def cost_compute(y_pred ,y_true):
        m = len(y_true)
        return -(1/m) * np.sum(y_true - np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0 
        y = y.reshape(-1, 1)
        for _ in range(self.epochs):

            Linear_eq = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(Linear_eq)

            error = y_predicted - y 
            dw =  (1/n_samples) * np.dot(X.T, error)
            db =  (1/n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate *  db

            cost = self.compute_cost(y, y_predicted)
            self.cost_history.append(cost)


    def predict(self, X):
        Linear_eq = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(Linear_eq) # will return the probabilites

if __name__ == "__main__":

    X = np.array([
        [2.0, 3.0],
        [1.0, 1.0],
        [3.0, 5.0],
        [5.0, 8.0],
        [8.0, 13.0],
        [10.0, 15.0]
    ])
    y = np.array([0, 0, 0, 1, 1, 1]) 


    model = LogisticRegression(learning_rate=0.01, epochs=1000)
    model.fit(X, y)


    predictions = model.predict(X)

    predictions = np.where(predictions >= 0.7, 1, 0)
    
    print("Predictions:", predictions)
    print("Final Cost:", model.cost_history[-1])  # Final cost after training
    print("Weights:", model.weights)
    print("Bias:", model.bias)

#this  is my branch

