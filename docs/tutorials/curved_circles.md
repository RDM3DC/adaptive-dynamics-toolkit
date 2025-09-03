# Curved Circles with Adaptive π

This tutorial demonstrates how the Adaptive π (πₐ) geometry works in practice by visualizing circles in curved space.

## Overview

In standard Euclidean geometry, π is constant (approximately 3.14159...). But in curved spaces, the ratio of a circle's circumference to its diameter can vary based on local curvature. The Adaptive Dynamics Toolkit provides tools for working with these curved geometries.

## Complete Example

The full example can be found in the [examples/pi_a_curved_circles.ipynb](https://github.com/RDM3DC/adaptive-dynamics-toolkit/blob/main/examples/pi_a_curved_circles.ipynb) notebook.

## Key Components

### Creating a Curvature Function

```python
import numpy as np
from adaptive_dynamics.pi import AdaptivePi

# Define a gentle curvature field (e.g., a Gaussian bump)
def k_field(x, y):
    return 0.25*np.exp(-((x**2 + y**2)/0.8)) - 0.05  # small negative baseline, positive bump near origin

# Create an AdaptivePi instance with this curvature field
pi_a = AdaptivePi(curvature_fn=k_field)
```

### Visualizing Circles in Curved Space

```python
import matplotlib.pyplot as plt

# Set up the plot
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.set_title('Curved vs Euclidean circles (center at origin)')

# Parameters for drawing circles
radius = 1.0
theta = np.linspace(0, 2*np.pi, 512)

# Draw Euclidean circle
ax.plot(radius*np.cos(theta), radius*np.sin(theta), 
        linestyle='--', label='Euclidean')

# Draw Adaptive circle (scaled by πₐ/π)
scale = pi_a.pi_a(0.0, 0.0)/np.pi
ax.plot(scale*radius*np.cos(theta), scale*radius*np.sin(theta), 
        label=f'Adaptive (scale={scale:.3f})')

ax.legend(loc='best')
plt.show()
```

## Understanding the Results

When the curvature is positive (`k > 0`):
- πₐ > π
- Circles appear larger than Euclidean circles (scale > 1)

When the curvature is negative (`k < 0`):
- πₐ < π
- Circles appear smaller than Euclidean circles (scale < 1)

In the flat limit (`k = 0`):
- πₐ = π
- Circles are identical to Euclidean circles

## Advanced Usage

For more accurate geometry calculations beyond the simple first-order approximation shown here:

1. Use the Gauss-Bonnet tools in `adaptive_dynamics.pi.gauss_bonnet`
2. Implement your own curvature model based on physical constraints
3. Check the [`geodesic_distance`](../api/pi.md) method for measuring distances in curved spaces