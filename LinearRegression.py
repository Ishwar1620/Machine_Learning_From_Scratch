import numpy as np

class LinearRegression:

    def __init__(self,learning_rate=0.01,  epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.cost_history = []

    def compute_cost(self, y_true, y_pred):

        m = len(y_true)

        return (1/(2*m)) * np.sum((y_true - y_pred)**2)

    def fit(self,X,y):
        n_smaples,n_features  = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            y_predicted = np.dot(X, self.weights) + self.bias

            #Gradient descent algorithm
            dw = (1/n_samples) * np.dot(X.T , (y_predicted - y ))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

            cost  = self.compute_cost(y, y_predicted)
            self.cost_history.append(cost)


    def predict(self,X):
        return np.dot(X,self.weights) + self.bias

if __name__ == "__main__":
    X = np.array([
        [1, 2],   # Feature 1, Feature 2
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6]
    ])
    y = np.array([5, 8, 11, 14, 17])

    model = LinearRegression(learning_rate=0.01, epochs=100)
    model.fit(X, y)

    predictions = model.predict(X)


    print("Predictions:", predictions)
    print("Final Cost:", model.cost_history[-1])  
    print("Weights:", model.weights)
    print("Bias:", model.bias)
