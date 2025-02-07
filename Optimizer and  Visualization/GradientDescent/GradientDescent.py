import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x**2 + 10 * np.sin(x)

def grad_f(x):
    return 2 * x + 10 * np.cos(x)

def gradient_descent(starting_point, learning_rate, num_iterations):
    x = starting_point
    history = [x]  # Store the history of x values
    for _ in range(num_iterations):
        gradient = grad_f(x)
        x = x - learning_rate * gradient
        history.append(x)
    return x, history

if __name__ == "__main__":
    starting_point = 0.0
    learning_rate = 0.1
    num_iterations = 50
    
    # Run Gradient Descent
    x_opt, history = gradient_descent(starting_point, learning_rate, num_iterations)
    
    # Visualization
    x_vals = np.linspace(-10, 10, 100)
    plt.plot(x_vals, f(x_vals), label="f(x)")
    plt.scatter(history, [f(x) for x in history], c="red", label="Gradient Descent Steps")
    plt.legend()
    plt.title("Gradient Descent Optimization")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()