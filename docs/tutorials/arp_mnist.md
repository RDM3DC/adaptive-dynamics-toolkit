# ARP Optimizer with MNIST

This tutorial demonstrates how to use the ARP optimizer in the Adaptive Dynamics Toolkit to train a neural network on the MNIST dataset.

## Overview

The ARP (Adaptive Resistance-Potential) optimizer is a novel optimization algorithm that adjusts learning dynamics based on a conductance state. It often provides better convergence properties than traditional optimizers like SGD or Adam in certain problem spaces.

## Complete Example

The full example can be found in the [examples/arp_mnist.ipynb](https://github.com/RDM3DC/adaptive-dynamics-toolkit/blob/main/examples/arp_mnist.ipynb) notebook.

## Key Components

### Setting up the Optimizer

```python
from adaptive_dynamics.arp.optimizers import ARP

# Define your PyTorch model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 256), nn.ReLU(),
    nn.Linear(256, 10)
).to(device)

# Create the ARP optimizer
opt = ARP(model.parameters(), 
          lr=3e-3,     # learning rate
          alpha=0.01,  # conductance growth rate
          mu=0.001)    # conductance decay rate
```

### Training Loop

```python
# Standard PyTorch training loop
for epoch in range(epochs):
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        logits = model(X)
        loss = criterion(logits, y)
        
        # Backward pass and optimization
        loss.backward()
        opt.step()
        opt.zero_grad()
```

## Tuning Parameters

The ARP optimizer has two key parameters that control its behavior:

- **alpha**: Controls how quickly conductance grows with gradient. Higher values make the optimizer more responsive.
- **mu**: Controls how quickly conductance decays over time. Higher values make the optimizer "forget" past gradients more quickly.

Try varying these parameters to see how they affect learning dynamics.

## Performance

ARP often performs well in problems with:

1. Highly curved loss landscapes
2. Noisy gradients
3. Need for adaptive step sizes

For a more detailed explanation of how ARP works internally, see the [ARP Optimizer API documentation](../api/arp.md).