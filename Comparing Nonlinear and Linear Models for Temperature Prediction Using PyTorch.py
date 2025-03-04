import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


t_u = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # Input temperatures in Celsius
t_c = torch.tensor([33.8, 35.6, 37.4, 39.2, 41.0])  # Corresponding temperatures in Fahrenheit


# Model definition
def model(t_u, w2, w1, b):
    return w2 * t_u ** 2 + w1 * t_u + b

# Loss function
def compute_loss(t_p, t_c):
    return torch.mean((t_p - t_c) ** 2)

# Training loop with prediction storage
def training_loop(epochs, learning_rate, model_params, t_u, t_c):
    w2, w1, b = model_params
    optimizer = optim.SGD([w2, w1, b], lr=learning_rate)
    losses = []

    for epoch in range(epochs):
        t_p = model(t_u, w2, w1, b)
        loss = compute_loss(t_p, t_c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Loss {loss.item()}')
        
        losses.append(loss.item())

    return losses, t_p

# Training with different learning rates
learning_rates = [0.1, 0.01, 0.001, 0.0001]
plt.figure(figsize=(12, 8))

for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    model_params = [torch.randn(1, requires_grad=True), torch.randn(1, requires_grad=True), torch.randn(1, requires_grad=True)]
    losses, predictions = training_loop(5000, lr, model_params, t_u, t_c)
    plt.plot(t_u, predictions.detach().numpy(), label=f'LR={lr}')

# Training linear model
linear_model_params = [torch.randn(1, requires_grad=True), torch.randn(1, requires_grad=True), torch.randn(1, requires_grad=True)]
print("\nTraining linear model:")
linear_losses, linear_predictions = training_loop(5000, 0.01, linear_model_params, t_u, t_c)
plt.plot(t_u, linear_predictions.detach().numpy(), label='Linear Model', linestyle='--')

plt.scatter(t_u, t_c, label='Actual Data', color='black')
plt.legend()
plt.xlabel('Input Temperature (Celsius)')
plt.ylabel('Output Temperature (Fahrenheit)')
plt.title('Nonlinear vs. Linear Temperature Prediction Models')
plt.show()
