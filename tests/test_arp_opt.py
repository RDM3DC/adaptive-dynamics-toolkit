"""Tests for adaptive_dynamics.arp.optimizers module."""

import pytest

try:
    import torch
    import torch.nn as nn

    from adaptive_dynamics.arp.optimizers import ARP
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestARPOptimizer:
    """Test suite for ARP optimizer."""
    
    def test_initialization(self):
        """Test that the optimizer initializes correctly."""
        model = nn.Linear(10, 1)
        optimizer = ARP(model.parameters())
        
        # Check that parameters are registered
        assert len(optimizer.param_groups) == 1
        assert len(optimizer.param_groups[0]['params']) == 2  # weight and bias
        
        # Check default hyperparameters
        assert optimizer.param_groups[0]['lr'] == 1e-3
        assert optimizer.param_groups[0]['alpha'] == 0.01
        assert optimizer.param_groups[0]['mu'] == 0.001
        assert optimizer.param_groups[0]['weight_decay'] == 0.0
    
    def test_conductance_state_initialization(self):
        """Test that conductance state is initialized correctly."""
        model = nn.Linear(10, 1)
        optimizer = ARP(model.parameters())
        
        # State should be empty initially
        for param in model.parameters():
            assert param not in optimizer.state
        
        # Initialize state by performing a step
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        criterion = nn.MSELoss()
        
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Now state should exist and have a 'G' tensor
        for param in model.parameters():
            assert param in optimizer.state
            assert 'G' in optimizer.state[param]
            assert optimizer.state[param]['G'].shape == param.shape
    
    def test_conductance_update(self):
        """Test that conductance is updated correctly during optimization."""
        model = nn.Linear(2, 1)
        
        # Initialize with deterministic values for testing
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 2.0]]))
            model.bias.copy_(torch.tensor([0.0]))
        
        optimizer = ARP(model.parameters(), alpha=0.1, mu=0.01)
        
        # Training data
        x = torch.tensor([[1.0, 1.0]])
        y = torch.tensor([[4.0]])  # Target output
        
        # First step
        output = model(x)
        loss = (output - y).pow(2).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Check that conductance was updated
        weight_G = optimizer.state[model.weight]['G']
        bias_G = optimizer.state[model.bias]['G']
        
        assert torch.all(weight_G > 0)
        assert torch.all(bias_G > 0)
        
        # Remember the first conductance values
        first_weight_G = weight_G.clone()
        
        # Second step with same input
        output = model(x)
        loss = (output - y).pow(2).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Conductance should be higher now due to accumulated adaptation
        assert torch.all(optimizer.state[model.weight]['G'] > first_weight_G)
    
    def test_reset_conductance(self):
        """Test that conductance can be reset."""
        model = nn.Linear(2, 1)
        optimizer = ARP(model.parameters())
        
        # Training data
        x = torch.tensor([[1.0, 1.0]])
        y = torch.tensor([[2.0]])
        
        # Do a step to initialize conductance
        output = model(x)
        loss = (output - y).pow(2).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Check that conductance exists
        for param in model.parameters():
            assert torch.any(optimizer.state[param]['G'] != 0)
        
        # Reset conductance
        optimizer.reset_conductance()
        
        # Check that conductance is zeroed
        for param in model.parameters():
            assert torch.all(optimizer.state[param]['G'] == 0)
            
    def test_weight_decay(self):
        """Test that weight decay is applied correctly."""
        # Create a model with non-zero weights
        model = nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0]]))
            model.bias.copy_(torch.tensor([0.0]))
        
        # Set a high weight decay for testing
        optimizer = ARP(model.parameters(), weight_decay=0.1)
        
        # Zero input should give zero gradients for bias
        x = torch.zeros(1, 1)
        y = torch.zeros(1, 1)
        
        output = model(x)
        loss = (output - y).pow(2).sum()
        loss.backward()
        
        # Store original weight
        original_weight = model.weight.clone()
        
        optimizer.step()
        optimizer.zero_grad()
        
        # With zero input and gradient but positive weight decay,
        # the weight should decrease in absolute value
        assert torch.abs(model.weight) < torch.abs(original_weight)