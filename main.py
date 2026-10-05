import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class SimpleLinearRegression:
    def __init__(self):
        self.slope = 0
        self.intercept = 0

    def fit(self, X, y):
        epochs = 1000
        learning_rate = 0.001

        n = len(X)

        self.history = [(self.slope, self.intercept)]
        self.mse_history = []

        for _ in range(epochs):
            y_pred = self.predict(X)

            error = y_pred - y

            mse = np.mean(error ** 2)
            self.mse_history.append(mse)

            dm = (2/n) * np.sum(error * X)
            db = (2/n) * np.sum(error)

            self.slope -= learning_rate * dm
            self.intercept -= learning_rate * db

            self.history.append((self.slope, self.intercept))


    def predict(self, X):
        return self.slope * X + self.intercept


np.random.seed(0)
X = np.linspace(0, 10, 100)
true_slope = 2.5
true_intercept = 1.5
y = true_slope * X + true_intercept + np.random.normal(0, 1, 100)

regression = SimpleLinearRegression()
regression.fit(X, y)

print(f"True slope: {true_slope}, Estimated slope: {regression.slope}")
print(f"True intercept: {true_intercept}, Estimated intercept: {regression.intercept}")

r_squared = 1 - (np.sum((y - regression.predict(X)) ** 2) / np.sum((y - np.mean(y)) ** 2))

print(f"R-squared: {r_squared}")

# Plot MSE
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(regression.mse_history)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training Loss')

#Animation Code For Regression
x_line = np.linspace(X.min(), X.max(), 100)
y_line = regression.predict(x_line)

fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='Data points')
line, = ax.plot([], [], color='red', label='Regression line')

ax.set_xlabel('X')
ax.set_ylabel('y')
ax.legend()

def animate(frame):
    slope, intercept = regression.history[frame]
    y_line = slope * x_line + intercept
    line.set_data(x_line, y_line)

ani = FuncAnimation(fig, animate, frames=len(regression.history), interval=100)

plt.show()
