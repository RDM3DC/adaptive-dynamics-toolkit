# Getting Started with Adaptive Dynamics Toolkit

This guide will help you set up and start using the Adaptive Dynamics Toolkit for your projects.

## Installation

### Basic Installation

Install ADT with pip:

```bash
pip install adaptive-dynamics
```

### Development Installation

For development or to include optional dependencies:

```bash
# Clone the repository
git clone https://github.com/RDM3DC/adaptive-dynamics-toolkit.git
cd adaptive-dynamics-toolkit

# Install with development dependencies
pip install -e ".[dev,docs,torch,sympy]"

# Alternatively with uv
uv venv && uv pip install -e ".[dev,docs,torch,sympy]"
```

## Core Components

ADT is organized into several modules:

### 1. Adaptive π (πₐ)

Working with geometry where π adapts to local curvature.

```python
from adaptive_dynamics.pi.geometry import AdaptivePi

# Define a curvature function (positive curvature example)
def curvature(x, y):
    return 0.01 * (x**2 + y**2)

# Create an AdaptivePi instance
pi = AdaptivePi(curvature_fn=curvature)

# Calculate πₐ at different points
pi_at_origin = pi.pi_a(0, 0)  # Standard π at origin
pi_at_point = pi.pi_a(1, 1)   # πₐ at point (1,1) with curvature

print(f"πₐ at origin: {pi_at_origin}")
print(f"πₐ at (1,1): {pi_at_point}")
```

### 2. ARP Optimization

The ARP optimizer can be used with PyTorch models:

```python
import torch
import torch.nn as nn
from adaptive_dynamics.arp.optimizers import ARP

# Define a model
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# Create an ARP optimizer
optimizer = ARP(
    model.parameters(),
    lr=1e-3,        # Learning rate
    alpha=0.01,     # Adaptation rate
    mu=0.001,       # Decay rate
    weight_decay=0  # L2 regularization
)

# Use in a training loop
for epoch in range(10):
    for x_batch, y_batch in data_loader:
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(x_batch)
        
        # Compute loss
        loss = nn.MSELoss()(y_pred, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Update weights with ARP
        optimizer.step()
```

### 3. Physics Simulations

ADT provides several simulation tools for physical systems:

```python
from adaptive_dynamics.sim.gravity import NBodySimulator

# Create a gravitational N-body simulator
sim = NBodySimulator(
    positions=[(0, 0, 0), (1, 0, 0)],  # Initial positions
    masses=[1.0, 0.01],                # Masses
    velocities=[(0, 0, 0), (0, 1, 0)]  # Initial velocities
)

# Run simulation
sim.run(time_step=0.01, total_time=10.0)

# Get final positions
final_positions = sim.get_positions()
```

## Next Steps

- Browse the [API documentation](./api/pi.md) for detailed references
- Try the [tutorials](./tutorials/curved_circles.md) for guided examples
- Contribute to the project on [GitHub](https://github.com/RDM3DC/adaptive-dynamics-toolkit)