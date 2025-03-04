# Predicting Housing Prices Using Linear Regression with PyTorch: A Feature-Driven Approach

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# Load the housing dataset
housing_data = pd.read_csv('Housing (1).csv')

# Extract input features (X) and target variable (y)
X = housing_data[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']].values
y = housing_data['price'].values

# Split the data into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input features (mean=0, std=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert NumPy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Reshape to (batch_size, 1)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# Loss function
criterion = nn.MSELoss()

# Training loop
def training_loop(epochs, learning_rate, model, X_train, y_train, X_val, y_val):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []

    for epoch in range(epochs): # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss Validation
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

    return train_losses, val_losses

# Training with different learning rates
learning_rates = [0.1, 0.01, 0.001, 0.0001]
best_model = None
best_val_loss = float('inf')

for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    model = LinearRegressionModel(input_size=5)  # 5 input features
    train_losses, val_losses = training_loop(5000, lr, model, X_train, y_train, X_val, y_val)

    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        best_model = model

# Pick the best model and report the final results
print("\nBest Linear Regression Model:")
print(f"Best Validation Loss: {best_val_loss}")


# Function to plot linear regression predictions
def plot_linear_regression(model, X, y, title):
    model.eval()
    with torch.no_grad():
        predictions = model(X).numpy()

    plt.scatter(y, predictions, alpha=0.5)
    plt.plot([min(y), max(y)], [min(y), max(y)], '--', color='red', linewidth=2)  
    plt.title(title)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.show()

# Plotting for the best model
plot_linear_regression(best_model, X_train, y_train, 'Training Set Linear Regression')
plot_linear_regression(best_model, X_val, y_val, 'Validation Set Linear Regression')

