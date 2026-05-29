import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model import SimpleSpectrogramCNN
from data_loader import AudioDataset

# Directories
REAL_DIR = "real/dir"
FAKE_DIRS = "fake/dir"

BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 50
SR = 16000
DURATION = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = AudioDataset(REAL_DIR, FAKE_DIRS, sr=SR, duration=DURATION)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, loss, optimizer
model = SimpleSpectrogramCNN(n_classes=1).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop with tqdm
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    # Wrap dataloader in tqdm to show percentage progress
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

    for i, (specs, labels) in enumerate(loop):
        specs, labels = specs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)

        # Skip batches with NaN loss
        if torch.isnan(loss):
            print(f"âš ï¸ NaN loss detected in batch {i}, skipping")
            continue

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Update tqdm description with running average loss
        loop.set_postfix(loss=running_loss / (i + 1))

    print(f"Epoch {epoch+1}/{EPOCHS} completed. Train Loss: {running_loss/len(train_loader):.4f}")

# Save trained model
torch.save(model.state_dict(), "model.pth")
print("âœ… Training complete. Model saved as model.pth")