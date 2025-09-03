# Adaptive Dynamics Toolkit Examples

This directory contains example notebooks demonstrating the core capabilities of the Adaptive Dynamics Toolkit.

## Available Examples

- **[πₐ Curved Circles](pi_a_curved_circles.ipynb)**: Demonstrates how the adaptive π (πₐ) geometry works with curved spaces.
- **[ARP Optimizer on MNIST](arp_mnist.ipynb)**: Shows how to train a simple neural network on MNIST using the ARP optimizer.

## Requirements

These examples require the optional dependencies of the toolkit:

```bash
pip install "adaptive-dynamics[torch,sympy]"
```

For the MNIST example, you'll also need `torchvision`:

```bash
pip install torchvision
```

## Notes

- The MNIST notebook downloads the dataset to `./data/` automatically. If you're offline, run it where you have connectivity first.
- The πₐ demo uses a toy correction formula `πₐ = π·(1 + 0.5·k)` for clarity. For more accurate results, use the Gauss-Bonnet machinery included in the toolkit.