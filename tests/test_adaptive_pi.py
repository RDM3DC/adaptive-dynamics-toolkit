"""Tests for adaptive_dynamics.pi.geometry module."""

import numpy as np

from adaptive_dynamics.pi.geometry import AdaptivePi


def test_flat_limit():
    """Test that in flat space, πₐ equals π exactly."""
    pi = AdaptivePi()
    assert np.allclose(pi.pi_a(0, 0), np.pi, rtol=1e-12)


def test_circumference_recovers_flat():
    """Test that circle circumference in flat space is 2πr."""
    pi = AdaptivePi(curvature_fn=lambda x, y: 0.0)
    r = 1.0
    assert np.isfinite(pi.circle_circumference(r))
    assert np.allclose(pi.circle_circumference(r), 2 * np.pi * r, rtol=1e-12)


def test_positive_curvature():
    """Test that positive curvature increases πₐ value."""
    flat_pi = AdaptivePi(curvature_fn=lambda x, y: 0.0)
    curved_pi = AdaptivePi(curvature_fn=lambda x, y: 1e-3)
    
    # πₐ should be larger in positive curvature
    assert curved_pi.pi_a(0, 0) > flat_pi.pi_a(0, 0)


def test_negative_curvature():
    """Test that negative curvature decreases πₐ value."""
    flat_pi = AdaptivePi(curvature_fn=lambda x, y: 0.0)
    curved_pi = AdaptivePi(curvature_fn=lambda x, y: -1e-3)
    
    # πₐ should be smaller in negative curvature
    assert curved_pi.pi_a(0, 0) < flat_pi.pi_a(0, 0)


def test_circle_area():
    """Test that circle area is πₐr² in curved space."""
    pi = AdaptivePi(curvature_fn=lambda x, y: 2e-3)
    r = 1.0
    
    pi_a_value = pi.pi_a(0, 0)
    expected_area = pi_a_value * r * r
    
    assert np.allclose(pi.circle_area(r), expected_area, rtol=1e-12)


def test_linear_distance():
    """Test that linear distance recovers Euclidean in flat space."""
    pi = AdaptivePi()
    x1, y1 = 0.0, 0.0
    x2, y2 = 3.0, 4.0
    expected = 5.0  # Pythagorean triple
    
    assert np.allclose(pi.linear_distance(x1, y1, x2, y2), expected, rtol=1e-12)