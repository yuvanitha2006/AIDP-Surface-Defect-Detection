import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from datasets.dataset_loader import ClimateTimeSeriesDataset
from models.lstm_autoencoder import LSTMAutoencoder


# =====================
# CONFIGURATION
# =====================
CSV_PATH = "data/raw/DailyClimateTrain.csv"
WINDOW_SIZE = 30
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 4
HIDDEN_DIM = 64
LATENT_DIM = 16
DROPOUT = 0.2

MODEL_SAVE_PATH = "models/lstm_autoencoder.pth"


# =====================
# DATA LOADERS
# =====================
train_dataset = ClimateTimeSeriesDataset(
    csv_path=CSV_PATH,
    window_size=WINDOW_SIZE,
    train=True
)

test_dataset = ClimateTimeSeriesDataset(
    csv_path=CSV_PATH,
    window_size=WINDOW_SIZE,
    train=False
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =====================
# MODEL
# =====================
model = LSTMAutoencoder(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM,
    dropout=DROPOUT
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# =====================
# TRAINING LOOP
# =====================
best_loss = np.inf
patience_counter = 0
train_losses = []

print(f"Training on device: {DEVICE}")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for batch in train_loader:
        batch = batch.to(DEVICE)

        optimizer.zero_grad()
        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.6f}")

    # ----- Early Stopping -----
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("Early stopping triggered.")
        break


# =====================
# PLOT TRAINING LOSS
# =====================
plt.figure()
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()


# =====================
# MC DROPOUT INFERENCE
# =====================
def mc_dropout_inference(model, dataloader, n_samples=30):
    model.train()  # Enable dropout
    all_scores = []
    all_uncertainties = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(DEVICE)
            recon_errors = []

            for _ in range(n_samples):
                recon = model(batch)
                error = torch.mean((batch - recon) ** 2, dim=(1, 2))
                recon_errors.append(error.unsqueeze(0))

            recon_errors = torch.cat(recon_errors, dim=0)
            mean_error = recon_errors.mean(dim=0)
            uncertainty = recon_errors.var(dim=0)

            all_scores.append(mean_error.cpu())
            all_uncertainties.append(uncertainty.cpu())

    return torch.cat(all_scores), torch.cat(all_uncertainties)


# =====================
# ANOMALY SCORING
# =====================
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
scores, uncertainties = mc_dropout_inference(model, test_loader)

scores = scores.numpy()
uncertainties = uncertainties.numpy()

threshold = np.percentile(scores, 95)
anomalies = scores > threshold

print(f"Anomaly threshold (95th percentile): {threshold:.6f}")
print(f"Detected anomalies: {np.sum(anomalies)} / {len(anomalies)}")


# =====================
# PLOT ANOMALIES
# =====================
plt.figure(figsize=(12, 4))
plt.plot(scores, label="Anomaly Score")
plt.plot(uncertainties, label="Uncertainty", alpha=0.7)
plt.axhline(threshold, color="red", linestyle="--", label="Threshold")
plt.legend()
plt.title("Anomaly Scores & Uncertainty")
plt.show()


