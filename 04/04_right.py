# 1. Initialize the weights
linear_model = nn.Linear(28 * 28, 1)

optimizer = torch.optim.SGD(linear_model.parameters(), lr=1)
epochs = 10

for epoch in range(epochs):
    # Train
    for xb, yb in dl:
        # 2. Forward pass, compute predictions
        preds = linear_model(xb)
        
        # 3. Compute loss
        loss = nn.BCEWithLogitsLoss()(preds, yb)
        

        # 4. Compute gradients
        loss.backward()
        
        # 5. Update weights
        optimizer.step()
        optimizer.zero_grad()