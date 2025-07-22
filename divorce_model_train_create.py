import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from model import DivorcePredictor  # Make sure this class matches the architecture you plan to use

# 1. Define data

data = pd.read_csv("marriage_data_india.csv")  # Replace with actual path

# One-hot encode all categorical columns
data_encoded = pd.get_dummies(data, drop_first=True)


# Set target column
target = 'Divorce_Status_Yes'  # Predict whether divorced (That column exists because of one-hot encoded divorce_status.)

# Features: everything except target
X = data_encoded.drop(columns=[target])
y = data_encoded[target]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Ensure all data is in numeric format (convert True/False to 1/0)
X_train = X_train.astype(float)
X_test = X_test.astype(float)
y_train = y_train.astype(float)
y_test = y_test.astype(float)

# Convert to PyTorch tensors (float32 for compatibility)
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # shape [N, 1]
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# 2. Define Model

input_dim = X_train_tensor.shape[1]
model = DivorcePredictor(input_dim)

# 3. Define Loss and Optimizer

criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train the Model

epochs = 100
for epoch in range(epochs):
    model.train()  # set to training mode

    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
    
    # Print progress
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# ==========================
# 5. Save the Trained Model
# ==========================
torch.save(model.state_dict(), "divorce_predictor_model.pth")
print("Model successfiu;lly saved to divorce_predictor_model.pth")
