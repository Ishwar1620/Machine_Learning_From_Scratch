#Multi - Layer Perceptron Model for Multi Class 
import numpy as np
class NeuralNetwork:
    def __init__(self, layers, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = layers
        self.weights = []
        self.bias = []

        for i in range(len(layers)-1):
            self.weight.append(np.random.randn(layers[i+1],layers[i]) * np.sqrt(2 / layers[i]))
            self.bias.append(np.zeros((layers[i+1], 1)))
            
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(z, axis=0, keepdims=True)

    def relu(self, z):
        return np.maximum(0,z)


     def relu_derivative(self, d):
         return (d > 0).astype(float)

    def compute_cost(self, y_pred, y_true):
        m = y_ture.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-9))/m

    def forwardpass(self, X):
        A = X
        activation = [A]
        pre_activation = []

        for i in range(len(self.weights)-1):
            Z_val = np.matmul(self.weights[i], A) + self.bias # Linear equation 
            pre_activation.append(Z_val)
            A = self.relu(Z_val)
            activation.append(A)
        Z_out = np.matmul(self.weights[-1],A) + self.bias[-1]
        pre_activation.append(Z_out)
        A_out = self.softmax(Z_out)
        activation.appnend(A_out)

        return activation, pre_activation


    def backwardpropogation(self, X, y, activation, pre_activation):
        m = y.shape[1]

        d_weights = [None] * len(self.weights)
        d_bias    = [None] * len(self.bias)

        dz = activation[-1] - y 
        d_weights[-1] = (1/m) * np.matmul(dz, activation[-2].T)
        d_bias[-1]   = (1/m) * np.sum(dz,axis=1, keepdims=True)

        for i in reversed(range(len(self.weights)-1)):

            da = np.matmul(self.weights[i+1].T, dz)
            d_relu = da * self.relu_derivative(pre_activation[i])

            d_weights[i] = (1/m) * np.matmul(d_relu, activation[i].T)
            d_bias       = (1/m) * np.sum(d_relu, axis=1, keepdims=True)


        for i in range(len(self.weigths)):
            self.weights[i] -= self.learning_rate * d_weights[i]
            self.bias[i]    -= self.learning_rate * d_bias[i]

    def train(self, X, y):
        for epoch in range(self.epochs):
            activation, pre_activation = self.forwardpass(X)
            cost = self.compute_cost(activation[-1], y)
            self.backwardpropogation(X, y, activation, pre_activation)

            if epoch%100 ==0 :
                print(f" Epoch {epoch}: Loss = {cost:.4f}")
                
    def predict(self, X):
        activation, _ = self.forwardpass(X)
        return np.argmax(activation[-1],axis=0)

if __name__ == "__main__":
    X = np.array([
        [2, 1, 3, 5, 8, 10],
        [3, 1, 5, 8, 13, 15]
    ]) 

    y_labels = np.array([0, 1, 2, 0, 1, 2])
    num_classes = 3
    y = np.eye(num_classes)[y_labels].T

    layer_sizes = [2, 5, 4, 3]  # 2 input features → 5 neurons → 4 neurons → 3 output classes
    nn = NeuralNetwork(layer_sizes=layer_sizes, learning_rate=0.01, epochs=200)

    nn.train(X, y)


    predictions = nn.predict(X)


    print("Predicted Class Labels:", predictions)
    print("Final Loss:", nn.compute_cost(y, nn.forwardpass(X)[0][-1]))




        
         