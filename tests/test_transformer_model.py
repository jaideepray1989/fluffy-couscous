import torch

from qd_suite.models.transformer_classifier import TransformerConfig, build_model


def test_transformer_model_creation():
    config = TransformerConfig(model_dim=64, nhead=2, num_layers=1)
    model = build_model(num_classes=10, config=config)
    assert model is not None


def test_transformer_model_forward_pass():
    config = TransformerConfig(model_dim=64, nhead=2, num_layers=1, include_time=False, deltas=True)
    model = build_model(num_classes=10, config=config)

    # Create a dummy input batch
    batch_size = 4
    seq_len = 20
    input_dim = 3  # dx, dy, pen
    
    x = torch.randn(batch_size, seq_len, input_dim)
    lengths = torch.randint(5, seq_len, (batch_size,))
    
    # Forward pass
    logits = model(x, lengths)
    
    # Check output shape
    assert logits.shape == (batch_size, 10)


def test_transformer_model_training_step():
    config = TransformerConfig(model_dim=64, nhead=2, num_layers=1, include_time=False, deltas=True)
    model = build_model(num_classes=10, config=config)
    
    # Create a dummy input batch
    batch_size = 4
    seq_len = 20
    input_dim = 3  # dx, dy, pen
    
    x = torch.randn(batch_size, seq_len, input_dim)
    lengths = torch.randint(5, seq_len, (batch_size,))
    labels = torch.randint(0, 10, (batch_size,))
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training step
    optimizer.zero_grad()
    logits = model(x, lengths)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0
