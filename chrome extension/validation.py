import torch
from torch.utils.data import DataLoader, random_split
from model import SimpleSpectrogramCNN
from data_loader import AudioDataset

# Directories
REAL_DIR = "real/dir"
FAKE_DIR = "fake/dir"

# Hyperparameters
BATCH_SIZE = 16
SR = 16000
DURATION = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = AudioDataset(REAL_DIR, FAKE_DIR, sr=SR, duration=DURATION)
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load model
model = SimpleSpectrogramCNN(n_classes=1).to(DEVICE)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.eval()

# Validation loop
correct, total = 0, 0
with torch.no_grad():
    for specs, labels in val_loader:
        specs, labels = specs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        outputs = model(specs)
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds.float() == labels).sum().item()
        total += labels.size(0)

print(f"✅ Validation Accuracy: {correct/total:.2%}")
