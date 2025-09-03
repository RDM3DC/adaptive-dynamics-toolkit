# Adaptive π (πₐ) API Reference

The `adaptive_dynamics.pi` module provides tools for working with geometry where π varies based on local curvature.

## AdaptivePi Class

::: adaptive_dynamics.pi.geometry.AdaptivePi

## Gauss-Bonnet Helpers

::: adaptive_dynamics.pi.gauss_bonnet.compute_geodesic_curvature

::: adaptive_dynamics.pi.gauss_bonnet.gauss_bonnet_integral

::: adaptive_dynamics.pi.gauss_bonnet.adaptive_pi_from_curvature

::: adaptive_dynamics.pi.gauss_bonnet.angle_sum_triangle

## Mathematical Background

The concept of Adaptive π (πₐ) is based on the relationship between curvature and geometry. In curved spaces, the properties we take for granted in Euclidean geometry, such as "π is always 3.14159...", no longer hold true.

### Key Concepts

1. **Local Value of π**: In curved spaces, π is no longer a constant but varies depending on location and curvature.

2. **Gauss-Bonnet Theorem**: Relates the total curvature of a surface to its topological characteristics, providing a foundation for calculating πₐ.

3. **Circumference and Area**: The ratio of a circle's circumference to its diameter (πₐ) changes with curvature, as does the relationship between radius and area.

### Example Calculations

On surfaces with constant curvature K:

- **Positive curvature** (like a sphere): πₐ < π
- **Flat space** (K = 0): πₐ = π
- **Negative curvature** (like a hyperbolic plane): πₐ > π

The first-order approximation used in the library is:

$$\pi_a \approx \pi \left(1 + \frac{K \cdot r^2}{6} + \ldots \right)$$

Where K is the Gaussian curvature and r is the radius of the circle.