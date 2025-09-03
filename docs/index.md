# Adaptive Dynamics Toolkit

<p align="center">
  <img src="assets/hero.gif" alt="Adaptive π visualization" width="640">
</p>

Welcome to the documentation for the Adaptive Dynamics Toolkit (ADT), a unified framework for adaptive computing paradigms.

## Overview

Adaptive Dynamics Toolkit integrates several novel computational approaches:

- **Adaptive π Geometry**: Work with curved spaces where π varies based on local geometry
- **ARP Optimization**: Neural network optimization using resistance-potential models
- **Physics Simulations**: Adaptive precision simulations for gravity, beams, and more
- **Compression Algorithms**: Text, curve, and tensor compression with adaptive precision
- **TSP & Slicing Tools**: 3D printing toolpath optimization using adaptive algorithms

## Installation

Install from PyPI:

```bash
pip install adaptive-dynamics
```

For development or with additional dependencies:

```bash
pip install "adaptive-dynamics[torch,sympy,dev]"
```

## Quick Examples

### Curved Geometry with Adaptive π (πₐ)

```python
from adaptive_dynamics.pi.geometry import AdaptivePi

# Create an instance with gentle positive curvature
pi = AdaptivePi(curvature_fn=lambda x, y: 1e-3)

# Calculate circumference in curved space
circumference = pi.circle_circumference(1.0)
print(f"Circumference of unit circle: {circumference:.6f}")
# Output: Circumference of unit circle: 3.144159
```

### Neural Network Training with ARP Optimizer

```python
import torch
import torch.nn as nn
from adaptive_dynamics.arp.optimizers import ARP

# Define a simple model
model = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 10))

# Use ARP optimizer
opt = ARP(model.parameters(), lr=3e-3, alpha=0.01, mu=0.001)

# Training loop (example)
# X, y = ... load a batch ...
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(model(X), y)
loss.backward()
opt.step()
opt.zero_grad()
```

## Project Status

ADT is currently in beta release (version 0.1.0). APIs may change before the 1.0 release.

## Contributing

We welcome contributions! See the [GitHub repository](https://github.com/RDM3DC/adaptive-dynamics-toolkit) for details on how to contribute.

## License

- Community Edition: [MIT License](https://github.com/RDM3DC/adaptive-dynamics-toolkit/blob/main/LICENSE)
- Pro Features: Commercial license