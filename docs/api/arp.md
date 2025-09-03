# ARP Optimizer API Reference

The `adaptive_dynamics.arp` module provides the ARP (Adaptive Resistance-Potential) optimization algorithm and related utilities.

## PyTorch Optimizer

::: adaptive_dynamics.arp.optimizers.ARP

## Core ODE System

::: adaptive_dynamics.arp.arp_ode.ARPSystem

## TensorFlow Implementation

::: adaptive_dynamics.arp.optimizers.TensorFlowARP

## Mathematical Background

The ARP (Adaptive Resistance-Potential) optimizer is based on principles inspired by electrical circuits, specifically the dynamics of resistors and conductors.

### Key Concepts

1. **Conductance State (G)**: A state variable that evolves over time based on gradient magnitudes, analogous to conductance in electrical systems.

2. **Core ODE**: The fundamental equation of ARP is:

   $$\frac{dG}{dt} = \alpha|I| - \mu G$$

   Where:
   - G is the conductance state
   - I is the input signal (gradient)
   - α is the adaptation rate
   - μ is the decay rate

3. **Learning Rate Modulation**: The effective learning rate is modulated by the conductance:

   $$\Delta w = -\frac{\eta \cdot \nabla L}{1 + G}$$

   Where:
   - η is the base learning rate
   - ∇L is the loss gradient
   - G is the conductance

### Advantages

- **Adaptive Step Sizes**: Automatically adjusts step sizes based on the history of gradient magnitudes
- **Stability**: Provides stability in regions of high gradient variability
- **Memory**: Maintains a "memory" of previous gradient patterns
- **No Momentum**: Achieves adaptive behavior without explicit momentum terms