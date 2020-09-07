# 1. Initialize the weights
weights = torch.randn((28 * 28, 1), requires_grad=True)
bias = torch.randn(1, requires_grad=True)
lr = 1
epochs = 10

for epoch in range(epochs):
    # Train
    for xb, yb in dl:
        # 2. Forward pass, compute predictions
        preds = xb@weights + bias
        
        # 3. Compute loss
        preds = preds.sigmoid().clamp(1e-6, 1 - 1e-6)
        loss = (-yb*torch.log(preds) - (1-yb)*torch.log(1-preds)).mean()
        
        # 4. Compute gradients
        loss.backward()
        
        # 5. Update weights
        with torch.no_grad():
            for p in [weights, bias]:
                p -= p.grad * lr
                p.grad.zero_()