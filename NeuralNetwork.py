#Multi - Layer Perceptron Model for Multi Class 
import numpy as np
import matplotlib.pyplot as plt
class NeuralNetwork:
    def __init__(self, layers, learning_rate, epochs,l2_lambda=0.1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = layers
        self.l2_lambda = l2_lambda
        self.weights = []
        self.bias = []
        self.cost_history=[]

        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[i+1],layers[i]) * np.sqrt(2 / layers[i]))
            self.bias.append(np.zeros((layers[i+1], 1)))
            
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def relu(self, z,alpha=0.01):
        return np.where(z > 0, z, alpha * z)


    def relu_derivative(self, d,alpha=0.01):
         return np.where(d > 0, 1, alpha)

    def compute_cost(self, y_pred, y_true):
        m = y_true.shape[1]
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
        l2_term = (self.l2_lambda / (2 * m)) * sum(np.sum(np.square(w)) for w in self.weights)
        return loss + l2_term

    def forwardpass(self, X):
        A = X
        activation = [A]
        pre_activation = []

        for i in range(len(self.weights)-1):
            Z_val = np.matmul(self.weights[i], A) + self.bias[i].reshape(-1, 1)# Linear equation 
            pre_activation.append(Z_val)
            A = self.relu(Z_val)
            activation.append(A)
        Z_out = np.matmul(self.weights[-1],A) + self.bias[-1]
        pre_activation.append(Z_out)
        A_out = self.softmax(Z_out)
        activation.append(A_out)

        return activation, pre_activation


    def backwardpropogation(self, X, y, activation, pre_activation):
        m = y.shape[1]

        d_weights = [None] * len(self.weights)
        d_bias    = [None] * len(self.bias)

        dz = activation[-1] - y 
        d_weights[-1] = (1/m) * np.matmul(dz, activation[-2].T) + (self.l2_lambda / m) * self.weights[-1]
        d_bias[-1]   = (1/m) * np.sum(dz,axis=1, keepdims=True)

        for i in reversed(range(len(self.weights)-1)):

            da = np.matmul(self.weights[i+1].T, dz)
            d_relu = da * self.relu_derivative(pre_activation[i])

            d_weights[i] = (1/m) * np.matmul(d_relu, activation[i].T) + (self.l2_lambda / m) * self.weights[i]
            d_bias[i]       = (1/m) * np.sum(d_relu, axis=1, keepdims=True)
            dz = d_relu


        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * d_weights[i]
            self.bias[i]    -= self.learning_rate * d_bias[i]

    def train(self, X, y):
        for epoch in range(self.epochs):
            activation, pre_activation = self.forwardpass(X)
            cost = self.compute_cost(activation[-1], y)
            self.cost_history.append(cost)
            self.backwardpropogation(X, y, activation, pre_activation)

            if epoch%100 ==0 :
                print(f" Epoch {epoch}: Loss = {cost:.4f}")
        #self.plot_loss_curve(self.cost_history)

    def plot_loss_curve(self,losses):
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(losses)), losses, label="Training Loss", color='b')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()
                
    def predict(self, X):
        activation, _ = self.forwardpass(X)
        return np.argmax(activation[-1],axis=0)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        y_true = np.argmax(y, axis=0)  # Convert one-hot labels to class indices
        return np.mean(y_pred == y_true) * 100


if __name__ == "__main__":
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y = make_classification(n_samples=6000, n_features=2, n_classes=3, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, class_sep=1.5, random_state=42)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000, random_state=42, stratify=y)
    

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert labels to one-hot encoding
    num_classes = 3
    y_train_onehot = np.eye(num_classes)[y_train].T
    y_test_onehot = np.eye(num_classes)[y_test].T
    layer_sizes = [2,32,16,3]  
    nn = NeuralNetwork(layers=layer_sizes, learning_rate=0.01, epochs=1000,l2_lambda=0.01)

    nn.train(X_train.T, y_train_onehot)


    predictions = nn.predict(X_train.T)

    print(nn.accuracy(X_train.T, y_train_onehot))
    print("Final Loss:", nn.compute_cost(y_train_onehot, nn.forwardpass(X_train.T)[0][-1]))
    

    predictions = nn.predict(X_test.T)
    accuracy = nn.accuracy(X_test.T, y_test_onehot)

    print(f"Accuracy on Test Data: {accuracy:.2f}%")


        
         